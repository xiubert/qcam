import numpy as np
import cv2
import matplotlib.pyplot as plt
from roipoly import RoiPoly
import joblib
import os
from glob import glob
import re
import math
import warnings

from holoviews import opts, streams
import holoviews as hv
import panel as pn
from matplotlib.path import Path

from lib.signalProcess import butterFilter, getTimeVec, dFFcalc
from lib.fileIngest import extract_qcamraw, qcams2imgs


"""
Functions for processing imaging signals.
"""

def calcSpatialBaseFluo(imgSeries: np.ndarray, t_baseAvg: tuple[float,float] = (1,3), **kwargs) -> np.ndarray:
    """
    Calculate spatial baseline fluorescence (at each pixel).

    Args:
        imgSeries (array): array of shape (Y, X, frame). assumed to be grayscale.
        t_baseAvg (tuple): start and end time points (inclusive) between which to calculate average of baseline fluorescence
        **kwargs: Optional arguments that will override default.

   Returns:
        spatialBaseFluo (numpy array): edge pixels take the value 255
    """
    # Optionally override parameters using kwargs
    t_baseAvg = kwargs.get('t_baseAvg', t_baseAvg)

    # get time array
    t = getTimeVec(imgSeries.shape[2], **kwargs)

    # Reshape to 2D: (number of pixels, time points)
    baselineIDX = np.where((t>=t_baseAvg[0]) & (t<=t_baseAvg[1]))[0]
    spatialBaseFluo = imgSeries[:,:,baselineIDX].mean(axis=2)

    return spatialBaseFluo

def calcSpatialDFFresp(imgSeries: np.ndarray, 
                        t_baseline: tuple[float,float] = (2,3),
                        stimlen: float = 0.4,
                        t_temporalAvg: tuple[float,float] = None,
                        temporalAvgFrameSpan: int = 10,
                        **kwargs) -> np.ndarray:
    """
    Calculate spatial dFF (dFF at each pixel).

    Args:
        imgSeries (array): array of shape (Y, X, frame). assumed to be grayscale.
        t_baseline (tuple): start and end time points (inclusive) of desired dFF baseline
        stimlen (float): length of stimulus (in seconds)
        t_temporalAvg (tuple): start and end time points (inclusive) between which to calculate average of spatialDFF
        temporalAvgFrameSpan (int): number of frames after stimulation for which spatialDFF average is computed
        **kwargs: Optional arguments that will override default.

   Returns:
        spatialDFFresp (numpy array): edge pixels take the value 255
    """
    # Optionally override parameters using kwargs
    stimlen = kwargs.get('stimlen', stimlen)
    temporalAvgFrameSpan = kwargs.get('temporalAvgFrameSpan', temporalAvgFrameSpan)
    t_temporalAvg = kwargs.get('t_temporalAvg', t_temporalAvg)

    # get time array
    t = getTimeVec(imgSeries.shape[2], **kwargs)

    # Reshape to 2D: (number of pixels, time points)
    reshaped_data = imgSeries.reshape(-1, imgSeries.shape[2])
    baselineIDX = np.where((t>=t_baseline[0]) & (t<=t_baseline[1]))[0]
    spatialbase = reshaped_data[:,baselineIDX].mean(axis=1).reshape(-1,1)
    spatialDFF = (reshaped_data-spatialbase)/spatialbase
    
    spatialDFF = butterFilter(spatialDFF,**kwargs)

    if t_temporalAvg is None:
        frameRate = kwargs.get('frameRate', 20)
        t_temporalAvg = (t_baseline[1]+stimlen,t_baseline[1]+stimlen+temporalAvgFrameSpan*(1/frameRate))

    spatialDFFresp = spatialDFF[:,np.where((t>=t_temporalAvg[0]) &
                                        (t<=t_temporalAvg[1]))[0]]\
                                            .mean(axis=1).reshape(*imgSeries.shape[:2])

    return spatialDFFresp


def getImgEdges(img: np.ndarray) -> np.ndarray:
    """
    Isolates edges of an image using Canny edge detection.

    Args:
        img (numpy array): image array (assume grayscale)
    Returns:
        edges (numpy array): edge pixels take the value 255
    """
    # 1: Normalize
    norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    # 2: Convert to grayscale (if needed)
    # gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY) if len(norm.shape) == 3 else norm
    # 3: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(norm, (5, 5), 0)
    # 4: Perform edge detection using Canny
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)  # Tune thresholds as needed

    return edges


def getXYdisp(Xcoor: list[np.ndarray], Ycoor: list[np.ndarray],
              frameDiff = 1):
    """
    Calculates difference in X and Y pixel coordinates across coordinate lists.

    Args:
        Xcoor (list): list of X coordinates (such as for blood vessel edge for each frame)
        Ycoor (list): list of Y coordinates (such as for blood vessel edge for each frame)
        frameDiff (int): number of frames between which to calculate the difference in pixel coordinates.

    Returns:
        Xdiffs (numpy array): average difference in pixel coordinates along X
        Ydiffs (numpy array): average difference in pixel coordinates along Y
    """

    Xdiffs,Ydiffs = [],[]
    for i,(Xidx,Yidx) in enumerate(zip(Xcoor[:-frameDiff],Ycoor[:-frameDiff])):
        # get X coordinates at same Y coordinates between frames
        a,b = np.intersect1d(Ycoor[i],Ycoor[i+frameDiff],return_indices=True)[1:]
        # get difference in X coordinates at same Y coordinates
        Xdiffs.append(np.mean(Xcoor[i+frameDiff][b]-Xidx[a]))
        # Same, but for Y
        a,b = np.intersect1d(Xcoor[i],Xcoor[i+frameDiff],return_indices=True)[1:]
        Ydiffs.append(np.mean(Ycoor[i+frameDiff][b]-Yidx[a]))
        
    return np.array(Xdiffs),np.array(Ydiffs)


def getEdgeXYdisp(imgSeries: np.ndarray, mask: np.ndarray, frameDiff: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the displacement of edge coordinates across frames of an image series, within a specified mask.

    Args:
        imgSeries (np.ndarray): 3D array of image frames with shape (height, width, frames).
        mask (np.ndarray): 2D binary mask array specifying the region of interest for edge detection.
        frameDiff (int): Number of frames between which to calculate the displacement in pixel coordinates.

    Returns:
        tuple:
            - Xdisp (np.ndarray): Array of average X-coordinate displacements between frames.
            - Ydisp (np.ndarray): Array of average Y-coordinate displacements between frames.
            - meanX (np.ndarray): Array of mean X-coordinates for each frame.
            - medianX (np.ndarray): Array of median X-coordinates for each frame.
            - meanY (np.ndarray): Array of mean Y-coordinates for each frame.
            - medianY (np.ndarray): Array of median Y-coordinates for each frame.

    Notes:
        - The function detects edges in each frame using `getImgEdges`.
        - Edge coordinates are constrained to the region defined by the mask.
        - Displacements are calculated by comparing pixel coordinates of edges across frames.
        - Mean and median coordinates are computed for both X and Y edge positions in each frame.

    Example:
        >>> Xdisp, Ydisp, meanX, medianX, meanY, medianY = getEdgeXYdisp(imgSeries, mask, frameDiff=2)
        >>> print("X displacements:", Xdisp)
        >>> print("Y displacements:", Ydisp)
        >>> print("Mean X-coordinates:", meanX)
        >>> print("Median X-coordinates:", medianX)

    """
    YXcoors = []
    for frame in np.arange(imgSeries.shape[2]):        
        edge = getImgEdges(imgSeries[:,:,frame])
        # get X and Y edge coordinates within mask
        YXcoors.append(np.where(mask*edge))

    # lists where each element is edge X or Y coordinate
    # length can differ depending on how many edge pixels identified
    Ycoors,Xcoors = zip(*YXcoors)

    # Calculates difference in X and Y pixel coordinates across coordinate lists.
    Xdisp,Ydisp = getXYdisp(Xcoors, Ycoors, frameDiff = frameDiff)

    # get mean and median of coordinates for absolute changes
    muX = np.array(list(map(lambda x: (np.median(x),np.mean(x)),Xcoors)))
    muY = np.array(list(map(lambda x: (np.median(x),np.mean(x)),Ycoors)))
    medianX,meanX = muX[:,0],muX[:,1]
    medianY,meanY = muY[:,0],muY[:,1]

    return Xdisp, Ydisp, meanX, medianX, meanY, medianY


def polygon2mask(points: list, image_shape: tuple = (130, 174), **kwargs):
    """
    Create a binary mask from polygon coordinates.

    This function generates a 2D binary mask by mapping a user-defined polygon onto an image grid.
    It determines which pixels fall inside the polygon and assigns them a value of `True` (1), 
    while the rest remain `False` (0). 

    Args:
        points (list): List containing coordinates of polygon vertices in tuple format (X, Y).
        image_shape (tuple, optional): Shape of the output binary mask as (height, width) or (Y, X). 
                                       Same as that of the image.
        **kwargs: Optional keyword arguments that can override `image_shape`.

    Returns:
        mask (numpy array): A 2D binary mask where pixels inside the polygon are `True`, 
                            and all other pixels are `False`.
    """
    
    # Optionally override parameters using kwargs
    image_shape = kwargs.get('image_shape', image_shape)

    # Check if any points exceed the image boundary
    for x, y in points:
        if x < 0 or x > image_shape[1] or y < 0 or y > image_shape[0]:
            warnings.warn(f"Point({x}, {y}) is out of bounds for image shape {image_shape}.")

    # Create empty binary mask
    mask = np.zeros(image_shape, dtype=bool)

    # Create a mesh grid of image coordinates
    y, x = np.mgrid[:image_shape[0], :image_shape[1]]

    # Combine coordinates into a flattened (N, 2) array
    coords = np.vstack((x.ravel(), y.ravel())).T

    # Create path object from polygon points
    path = Path(points)

    # Check which points are inside the polygon
    mask_flat = path.contains_points(coords)
    mask = mask_flat.reshape(image_shape)

    return mask


def getSquareMask(Xcoor: float, Ycoor: float, width: float, height: float, 
                 angle: float = 90, X_parallel : bool = True, **kwargs):
    """
    Manually generate a binary mask for a square- or parallelogram-shaped region of interest (ROI).

    This function creates a binary mask for a square or parallelogram with specified width, height, and an 
    adjustable angle. At least one pair of opposite sides should be parallel to either the X-axis or the Y-axis.
    The mask can be used to isolate specific regions within an image.

    Args:
        Xcoor (float): X-coordinate of the top-left vertex.
        Ycoor (float): Y-coordinate of the top-left vertex.
        width (float): Distance between left and right sides. Must be > 0.
        height (float): Distance between top and bottom sides. Must be > 0.
        angle (float, optional): Angle (in degrees) of the top-left vertex. Should be within range 0 to 180.
                                 Defaults to 90 degrees (creates a rectangle).
        X_parallel (bool, optional): Whether the parallelogram is aligned with the X-axis or Y-axis.
                                    - `True`: The top and bottom sides are parallel to the X-axis.
                                    - `False`: The left and right sides are parallel to the Y-axis.
                                    Defaults to `True`.
        **kwargs: Optional keyword arguments.

    Returns:
        mask_output (dict): A dictionary containing:
                            - `'mask'`: A 2D binary mask representing the parallelogram.
                            - `'ROIcontour'`: A NumPy array of the parallelogram's vertex coordinates, 
                                              including a repeated first vertex to close the shape.

    Notes:
        - The parallelogram is defined by four vertices, calculated using trigonometric functions.
        - The function internally calls `polygon2mask` to create the binary mask.
        - The contour array (`ROIcontour`) ensures the shape is closed by repeating the first vertex.
    """
    
    if width > 0 and height > 0:
        # Calculate vertex coordinates based on width, height, and angle
        if 0 < angle < 180:
            if X_parallel:
                # If the top and bottom sides are parallel to the X-axis
                shift_x = height/math.tan(math.radians(angle))
                points = [(Xcoor, Ycoor),
                          (Xcoor + width, Ycoor),
                          (Xcoor + width + shift_x, Ycoor + height),
                          (Xcoor + shift_x, Ycoor + height)]
            else:
                # If the left and right sides are parallel to the Y-axis
                shift_y = width/math.tan(math.radians(angle))
                points = [(Xcoor, Ycoor),
                          (Xcoor + width, Ycoor + shift_y),
                          (Xcoor + width, Ycoor + height + shift_y),
                          (Xcoor, Ycoor + height)]
        else:
            # Raise error for invalid angle degrees
            raise ValueError("`angle` must be between 0 and 180 degrees.")
    else:
        # Raise error for negative values of width or height
        raise ValueError("`width` and `height` must be positive values.")
    
    # Create binary mask from vertex coordinates
    mask = polygon2mask(points, **kwargs)

    # Convert the polygon vertices into a NumPy array
    contour = np.array(points)

    # Close the polygon by adding the first point at the end
    contour = np.vstack([contour, contour[0,:]])

    # Store mask and contour data in a dictionary
    mask_output = {'mask': mask, 'ROIcontour': contour}

    return mask_output


def getROImaskUI(image: np.ndarray, show_mask: bool = True, 
                 expDir: str = None, saveName: str = "response_mask", 
                 **kwargs):
    """
    Creates an interactive interface for defining a polygonal Region of Interest (ROI) on an image 
    and generates a corresponding binary mask. Optionally saves the mask as a `joblib` file in the designated directory.

    This function uses Holoviews and Panel to provide an interactive tool for drawing a polygon 
    on a given image. It generates a binary mask corresponding to the drawn polygon, 
    optionally displays the mask for visual confirmation and saves the mask as a `joblib` file.

    Args:
        image (numpy.ndarray): 2D array representing the input image on which the ROI is drawn.
        show_mask (bool): If True, displays the binary mask of the ROI after it is created.
        expDir (str, optional): Directory (folder) where the binary mask is to be saved.
                                - `None`: The binary mask cannot be saved. The `Save ROI mask` button is grey and cannot be clicked.
                                Defaults to `None`.
        saveName (str, optional): Name of the saved `joblib` file.
                                Defaults to `'response_mask'`.
        **kwargs: Optional key word arguments.

    Returns:
        tuple:
            - pn.Column: A Panel layout object containing the interactive interface 
              (image display, polygon tool, buttons, and mask output visualization).
            - dict: A dictionary containing the generated binary mask under the key `'mask'`, key `'ROIcontour'` for ROI points.

    Notes:
        - The ROI is defined interactively by drawing a polygon using the Holoviews `PolyDraw` tool.
        - The mask is created by mapping the polygon coordinates to the image dimensions.
        - The binary mask is stored in the returned dictionary as a 2D NumPy array.
        - The mask is flipped vertically to match the visualization orientation in Matplotlib.
        - The existing file of the same name in the same directory is backed up automatically to avoid overriding.

    Example:
        >>> import numpy as np
        >>> from getROImaskUI import getROImaskUI
        >>> image = np.random.rand(100, 100)  # Example image
        >>> panel_layout, mask_data = getROImaskUI(image)
        >>> panel_layout.show()  # Launch the interactive tool
        >>> # After drawing the ROI and clicking the button:
        >>> roi_mask = mask_data['mask']
        >>> print("Generated mask shape:", roi_mask.shape)

    Dependencies:
        - Holoviews (`hv`)
        - Panel (`pn`)
        - Matplotlib (`plt`)
        - NumPy
        - re
        - Shapely's `Path` for polygon operations

    Interactive Tools:
        - Polygon drawing is handled by Holoviews' `PolyDraw` tool.
        - `Get ROI mask` button to generate and display the binary mask.
        - `Save ROI mask` button to save the binary mask as a `joblib` file.

    Limitations:
        - Only one polygon can be drawn per invocation (due to `num_objects=1`).
        - Assumes the input image is grayscale or 2D.

    """
    # Optionally override parameters using kwargs
    show_mask = kwargs.get('show_mask', show_mask)
    expDir = kwargs.get('expDir', expDir)
    saveName = kwargs.get('saveName', saveName)

    # for running in jupyter notebook
    pn.extension()
    hv.extension('bokeh')

    # Initialize Holoviews objects
    image_hv = hv.Image(image, bounds=(0, 0, image.shape[1], image.shape[0])).opts(
        width=image.shape[1] * 3, 
        height=image.shape[0] * 3,
        cmap='jet',
    )

    poly = hv.Polygons([])
    poly_stream = streams.PolyDraw(source=poly, drag=True, num_objects=1, show_vertices=True)

    # Create dynamic polygon element
    dmap = hv.DynamicMap(lambda data: poly.clone(data), streams=[poly_stream])

    plot = (image_hv * poly).opts(
        opts.Polygons(fill_alpha=0.3, active_tools=['poly_draw'])
    )

    # Create button to get polygon coordinates and generate mask
    mask_output = {'mask': None}  # To store the generated mask
    output_pane = pn.pane.Matplotlib(height=300)  # To display the mask

    def get_coords(event):
        if poly_stream.data:
            
            coords = poly_stream.data
            # Convert coordinates to points format
            points = list(zip(coords['xs'][0], coords['ys'][0]))  # Assumes one polygon
           

            # Create and display mask
            mask = polygon2mask(points, image_shape=image.shape)
            
            # debug
            # print("Poly Stream Data:", poly_stream.data)
            # print("Extracted Points:", points)
            # print("Mask Created: Shape:", mask.shape, "Sum of Mask:", mask.sum())

            # Flip the mask vertically for correct display with matplotlib
            mask_flipped = np.flipud(mask)
            mask_output['mask'] = mask_flipped  # Store the mask in the result dictionary
            contour = np.array(points)
            contour[:,1] = mask_flipped.shape[0]-contour[:,1] #flip y values
            # close polygon
            contour = np.vstack([contour,contour[0,:]])
            mask_output['ROIcontour'] = contour

            # Visualize
            if show_mask:
                # plt.close('all')
                # plt.figure()
                # plt.imshow(mask_flipped, cmap='gray')
                # plt.title('ROI Mask')
                # plt.xlabel('X')
                # plt.ylabel('Y')
                # plt.show()
                fig, ax = plt.subplots()
                ax.imshow(mask_flipped, cmap='gray')
                plt.close(fig)
                ax.set_title("ROI Mask")
                output_pane.object = fig  # Update the output pane with the new plot

    # Button to generate the mask
    mask_button = pn.widgets.Button(name='Get ROI mask', button_type='primary')
    mask_button.on_click(get_coords)

    def save_mask(event):
        if mask_output['mask'] is not None:
            if expDir is not None:
                savePath = os.path.join(expDir, f"{saveName}.joblib")
                print(f"Save Path: {savePath}")
            else:
                print("Experiment directory not specified.")

            # Backup existing file
            if os.path.exists(savePath):
                print("File existing in the same path.")
                exist_files = glob(savePath.replace('.joblib', '_[0-9][0-9][0-9].joblib'))

                # Increment the maximum sequence number by 1 as backup path
                # seq_list = []
                # for file in exist_files:
                #     seqNum = re.search(r"_(\d{3})\.joblib$", file)
                #     seq_list.append(int(seqNum.group(1)))
                # seq_max = max(seq_list) if seq_list else 0
                seq_numbers = [
                    int(match.group(1)) for f in exist_files if (match := re.search(r"_(\d{3})\.joblib$", f))
                ]
                seq_max = max(seq_numbers) if seq_numbers else 0

                backupPath = savePath.replace('.joblib', f"_{seq_max + 1:03}.joblib")
                os.rename(savePath, backupPath)
                print(f"Existing files backed up to: {backupPath}")
                         
            # Save the new mask
            try:
                joblib.dump(mask_output['mask'], savePath)
                print(f"Mask saved successfully to: {savePath}")
            except Exception as e:
                print(f"Failed to save mask: {e}")
        else:
            print("No mask to save.")
    
    # Button to save the mask
    save_button = pn.widgets.Button(name='Save ROI mask', button_type='success', disabled=(expDir is None))
    save_button.on_click(save_mask)

    # Create the Panel layout
    layout = pn.Column(plot, pn.Row(mask_button, save_button), output_pane)

    # Return the Panel layout and allow access to the mask
    # return pn.Column(plot, button), mask_output
    return layout, mask_output


def qcams2roiTrace(qcams: list, baseline : bool = False, **kwargs):
    """
    Processes a list of qcam file paths to generate an interactive UI for drawing an ROI 
    and calculates the corresponding spatial Delta F/F (dFF) response or baseline fluorescence for the average image series.
    
    This function extracts images from the provided qcam paths, computes the average image series,
    calculates the spatial dFF response or baseline fluorescence, and returns an interactive user interface (UI) 
    for drawing a Region of Interest (ROI) on the computed spatial dFF response or baseline fluorescence.

    Args:
        qcams (list): A list of file paths to qcam files (e.g., '*.qcamraw').
        baseline (bool, optional): Whether to draw ROI on a spatial baseline fluorescence heatmap.
                                    - 'True': Interactive UI for ROI drawing is the spatial baseline fluorescence heatmap.
                                    - 'False': Interactive UI for ROI drawing is the spatial dFF response heatmap.
                                    Defaults to 'False'.
        **kwargs: Optional key word arguments.

    Returns:
        tuple:
            - UI: An interactive Panel layout for drawing an ROI on the spatial dFF response or baseline fluorescence.
            - mask_output (dict): A dictionary containing the binary mask of the drawn ROI.
            - imgs (numpy.ndarray): A 3D NumPy array of images extracted from the qcam files.
            - spatialDFF (numpy.ndarray): The spatial dFF response or baseline fluorescence (if baseline == True) for the average image series.

    Notes:
        - The function assumes that the qcam files are in a format supported by `qcams2imgs` for image extraction.
        - The spatial dFF response or baseline fluorescence is calculated on the mean of the extracted images across frames.
        - The ROI mask can be drawn interactively using the provided UI, and the generated mask can be used for further analysis.

    Example:
        >>> qcam_paths = ['path/to/qcam1.qcamraw', 'path/to/qcam2.qcamraw']
        >>> ui, mask_output, imgs, spatial_dff = qcams2roiTrace(qcam_paths)
        >>> ui  # Launch the interactive UI for ROI drawing
        >>> roi_mask = mask_output['mask']  # Access the generated ROI mask

    Dependencies:
        - `qcams2imgs`: A function for extracting images from qcam file paths.
        - `calcSpatialBaseFluo`: A function for calculating the spatial baseline fluorescence from the average image series.
        - `calcSpatialDFFresp`: A function for calculating the spatial Delta F/F response from the average image series.
        - `getROImaskUI`: A function that creates an interactive UI for drawing a polygon ROI and generates the binary mask.

    """
    imgs,_ = qcams2imgs(qcams)
    avgImgSeries = np.array(imgs).mean(axis=(0))
    
    if baseline:
        spatialDFF = calcSpatialBaseFluo(avgImgSeries, **kwargs)
    else:
        spatialDFF = calcSpatialDFFresp(avgImgSeries, **kwargs)
    ui, mask_output = getROImaskUI(spatialDFF, **kwargs)

    return ui, mask_output, np.array(imgs), spatialDFF


def mask2trace(mask: np.ndarray, imgs: np.ndarray, spatialDFF: np.ndarray = None, ROIcontour: np.ndarray = None, **kwargs):
    """
    Applies mask (of shape [Y, X]) to array of images of shape [trace, Y, X, frame]
    and returns average rawF within ROI across frames in array of shape [trace, frame]

    Args:
        mask (np.ndarray): 2D binary mask array specifying the region of interest for edge detection (shape: [Y, X]).
        imgs (numpy array): array of images of shape [trace, Y, X, frame]
        spatialDFF (numpy array): Spatial dFF response, shape [Y, X]
        ROIcontour (numpy array): [X, Y] coordinates of ROI
        kwargs: optional arguments for flexibility.
    Returns:
        ROITrace (numpy array): average fluorescence within ROI for each trace (shape: [trace, frame])
    """
    
    t = getTimeVec(imgs.shape[-1], **kwargs)
    ROItrace = imgs[:,mask==1,:].mean(axis=1)
    fig,ax = plt.subplots(3,2,figsize=(12,8))
    ax[0,0].imshow(imgs.mean(axis=(0,3)),cmap='gray')
    heatmapImg = ax[0,1].imshow(spatialDFF,cmap='jet')
    if isinstance(ROIcontour,np.ndarray):
        ax[0,0].plot(ROIcontour[:, 0], ROIcontour[:, 1], color='white', linewidth=2)
        ax[0,1].plot(ROIcontour[:, 0], ROIcontour[:, 1], color='black', linewidth=2)
    plt.colorbar(heatmapImg)
    
    # raw F over entire image
    ax[1,0].plot(t,imgs.mean(axis=(0,1,2)))
    ax[1,0].set_xlabel('time (s)')
    ax[1,0].set_ylabel('raw F (full img)')
    
    # raw F within ROI
    ax[1,1].plot(t,ROItrace.mean(axis=(0)))
    ax[1,1].set_title('roi')
    ax[1,1].set_xlabel('time (s)')
    ax[1,1].set_ylabel('raw F (ROI)')

    
    # dF and dFF
    dFF,dF,_ = dFFcalc(ROItrace.mean(axis=(0)),**kwargs)
    ax[2,0].plot(t,dF)
    ax[2,0].set_title('roi')
    ax[2,0].set_xlabel('time (s)')
    ax[2,0].set_ylabel('dF (ROI)')
    ax[2,1].plot(t,dFF)
    ax[2,1].set_title('roi')
    ax[2,1].set_ylabel('dFF (ROI)')
    ax[2,1].set_xlabel('time (s)')

    plt.tight_layout()
    fig.show()

    return ROItrace


def getROImask(imgPath: str = None,
               qcamFiles: list = None, 
               processEdges: bool = False, processResponse: bool = False,
               saveMask: bool = True, **kwargs) -> np.ndarray:
    """
    Captures ROI mask via user drawn polygon.

    **DEPRECATED** does not work in jupyter notebook.
    """
    if qcamFiles is not None:
        imgs = qcams2imgs(qcamFiles)[0]
        # take average across files
        imgSeries = np.array(imgs).mean(axis=0)
        # then avg across time
        img = imgSeries.mean(axis=2)
        savePath = ''
        saveMask=False
    
    else:
        if ".qcamraw" in imgPath and os.path.exists(imgPath):
            print("Showing temporal average of single qcamraw file for drawing ROI")
            imgSeries = extract_qcamraw(imgPath)[0]
            # take average across time
            img = imgSeries.mean(axis=2)
            savePath = imgPath.replace('.qcamraw','_mask.joblib')

        elif os.path.exists(imgPath):
            print("Showing temporal average of all .qcamraw files in directory for drawing ROI")
            qFiles = glob(os.path.join(imgPath,'*.qcamraw'))
            if len(qFiles)<1:
                ValueError("No .qcamraw files found in directory")
            imgs = qcams2imgs(qFiles)[0]
            # take average across files
            imgSeries = np.array(imgs).mean(axis=0)
            # then avg across time
            img = imgSeries.mean(axis=2)
            savePath = os.path.join(imgPath,'avgImg_mask.joblib')
            
        else:
            raise ValueError("Path does not exist")
    
    if not (processEdges ^ processResponse):
        return ValueError("Only one of processEdges or processResponse may be true")
    
    if processEdges:
        savePath = savePath.replace('_mask','_edge_mask')
        #Process frame for edges and visualize
        imgEdge = getImgEdges(img)
        plt.imshow(imgEdge+img*0.3, cmap='gray')

    elif processResponse:
        savePath = savePath.replace('_mask','_response_mask')
        t = getTimeVec(imgSeries.shape[2])

        # eventually use kwargs for calcSpatialDFFresp params
        spatialRespImg = calcSpatialDFFresp(imgSeries,
                            stimlen=0.1, temporalAvgFrameSpan=8)
        plt.imshow(spatialRespImg)
    else:
        # Display image
        plt.imshow(img, cmap='gray')

    # Prompt to draw a polygonal ROI
    print("Draw polygon... Double click to finish")
    roi = RoiPoly(color='r')

    # Get ROI mask
    mask = roi.get_mask((imgEdge if processEdges else img))
    
    if saveMask:
        if os.path.exists(savePath):
            mask_files = glob(savePath.replace('mask.joblib','mask*'))
            os.rename(savePath,savePath.replace('.joblib',f"_{len(mask_files):03}.joblib"))

        # save where image is saved
        joblib.dump(mask,savePath)

    # Display the ROI
    if processEdges:
        plt.imshow((img*.3+imgEdge)*(mask+0.6), cmap='gray')
        plt.show()

    if processResponse:
        _, ax = plt.subplots(1, 2, figsize=(12,6))
        ax[0].imshow(img, cmap='gray')
        ax[1].imshow(spatialRespImg*(mask+0.5), cmap='viridis')
        plt.show()
        
    else:
        plt.imshow(img*(mask+0.6), cmap='gray')
        plt.show()

    return mask,imgSeries





