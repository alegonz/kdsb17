import os
import warnings
from glob import glob
import dicom
import numpy as np
from skimage import filters, measure
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import binary_opening, binary_dilation


# Read sequence of dicom files of a patient
def read_dcm_sequence(patientid, base_path):
    """Read dicom files in path of a patient.
    
    Args:
        patientid (str): ID of the patient (patient foldername).
        base_path (str): Path containing the patient folder.
    
    Returns:
        A list of dicom slice data.
    """
    
    patient_path = os.path.join(base_path, patientid, '*.dcm')
    paths = glob(patient_path)
    
    return [dicom.read_file(path) for path in paths]


def check_sequence(dcm_seq, tol=1e-2):
    """Checks for each slice in the sequence the following:
    
    - Modality = 'CT'
    - BitsAllocated = 16
    - PhotometricInterpretation = 'MONOCHROME2'
    - Rows = 512
    - Columns = 512
    - SamplesPerPixel = 1
    - ImageOrientationPatient = [1,0,0,0,1,0]
    - Maximum pixel array value is less than 4096 if BitsStored = 12.
    - SeriesDescription = 'Axial'
    - RescaleType = '', 'HU' or 'US'
    - Overlap of scanned area between acquisitions, if any.
    - Checks if pixel spacing is uniform across slices.
    
    Args:
        dcm_seq (list): Sequence of dicom slice data.
        tol (float): Spacing uniformity tolerance in mm.
    
    Returns:
        Nothing. If overlap between acquisitions is found
        it keeps only the slices of the acquisition with
        the most slices (the list is modified in-place).
    
    """

    exception_template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    
    # Check basic requirements
    for dcm in dcm_seq:
        
        # These are Type 1 elements
        if dcm.Modality != 'CT':
            warnings.warn('Expected Modality=CT, got %s.' % dcm.Modality)

        if dcm.BitsAllocated != 16:
            warnings.warn('Expected BitsAllocated=16, got %d.' % dcm.BitsAllocated)

        if dcm.PhotometricInterpretation != 'MONOCHROME2':
            warnings.warn('Expected PhotometricInterpretation=MONOCHROME2, got %s.' % dcm.PhotometricInterpretation)

        if dcm.Rows != 512:
            warnings.warn('Expected Rows=512, got %d.' % dcm.Rows)

        if dcm.Columns != 512:
            warnings.warn('Expected Columns=512, got %d.' % dcm.Columns)

        if dcm.SamplesPerPixel != 1:
            warnings.warn('Expected SamplesPerPixel=1, got %d.' % dcm.SamplesPerPixel)

        if dcm.ImageOrientationPatient != [1, 0, 0, 0, 1, 0]:
            warnings.warn('Expected ImageOrientationPatient=[1,0,0,0,1,0], got %s.' % dcm.ImageOrientationPatient)
        
        if dcm.BitsStored == 12 and dcm.pixel_array.max() > 4095:
            warnings.warn('Pixel array values exceed 4095, but BitsStored=12.')
    
    # Check if overlapping acquisitions exist
    # Edge case with two overlapping acquisitions: b8bb02d229361a623a4dc57aa0e5c485
    positions = {}
    for dcm in dcm_seq:
        try:
            acq = dcm.AcquisitionNumber
        except AttributeError as ex:
            message = exception_template.format(type(ex).__name__, ex.args)
            warnings.warn(message)
            acq = None
        
        if acq in positions:
            positions[acq].append(float(dcm.ImagePositionPatient[2]))
        else:
            positions[acq] = [float(dcm.ImagePositionPatient[2])]
    
    if len(positions) > 1:
        intervals = [(acq, min(p), max(p), len(p)) for acq, p in positions.items()]
        intervals.sort(key=lambda interval: interval[1])
        
        (_, _, maxi_prev, _) = intervals[0]
        for (acq, mini, maxi, _) in intervals[1:]:
            if mini <= maxi_prev:
                print('Found overlap at acquisition', acq)
                
                intervals.sort(key=lambda interval: interval[3])
                acq = intervals[-1][0]
                
                print('Keeping acquisition', acq)
                
                dcm_seq = [d for d in dcm_seq if d.AcquisitionNumber == acq]
                
                break
            
            else:
                maxi_prev = maxi
    
    # Sort slices by z-axis position
    dcm_seq.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    # Check if pixel spacing if uniform across slices.
    z = np.float32([dcm.ImagePositionPatient[2] for dcm in dcm_seq])
    
    dz_std = np.diff(z).std()
    dy_std = np.float32([dcm.PixelSpacing[0] for dcm in dcm_seq]).std()
    dx_std = np.float32([dcm.PixelSpacing[1] for dcm in dcm_seq]).std()
    
    if (dz_std > tol) or (dy_std > tol) or (dx_std > tol):
        warnings.warn('Pixel spacing is not uniform.')

    return dcm_seq


# Convert raw dicom pixel values to array in Hounsfield Units.
def dcm2array(dcm):
    """Converts the raw pixel array in dicom data to an image array in HU units.
    Args:
        dcm (dicom.dataset.FileDataset): input dicom data.
    
    Returns:
        Image array in HU units.
    """
    exception_template = "An exception of type {0} occurred. Arguments:\n{1!r}"

    # Apparently pydicom is already taking into consideration
    # the PixelRepresentation and BitsAllocated when making the pixel_array.
    # However is not considering BitsStored nor HighBit.
    array = dcm.pixel_array

    # These are Type 1 so they shouldn't raise an error
    slope = dcm.RescaleSlope
    intercept = dcm.RescaleIntercept
    representation = dcm.PixelRepresentation

    # This is Type 3, so it might not be present in the dicom file
    try:
        padding = dcm.PixelPaddingValue
    except AttributeError as ex:
        message = exception_template.format(type(ex).__name__, ex.args)
        warnings.warn(message)
        padding = None

    # Determination of proper pixel padding value
    if isinstance(padding, int) and padding > 32767 and representation == 1:
        padding = padding.to_bytes(2, byteorder='little', signed=False)

    if isinstance(padding, bytes):
        padding = int.from_bytes(padding, byteorder='little', signed=(representation == 1))
    
    # Set padded area to air (HU=-1000)
    if padding is not None:
        array[array == padding] = -1000 - intercept

    # Safety measure
    # There a few cases in which the pixel padding value is valid
    # but does not correspond with the actual padded values in the data (< -1000).
    # Furthermore, CT is represented at most with 12 bits, thus it cannot exceed 4095.
    array[array <= -1000] = -1000 - intercept
    array[array > 4095] = 4095

    # Transform to Hounsfield Units.
    array = np.float64(array)
    array *= slope
    array += intercept
    array = np.int16(array)

    return array


# Stack dicom slice sequence into 3D array
def make_3d_array(dcm_seq):
    """Stacks sequence of dicom slices into a 3D array.
    
    Args:
        dcm_seq (list of dicom.dataset.FileDataset): Sequence of dicom slice data.
    
    Returns:
        A tuple containing:
            - A 3D array with the stacked slices
            - A list with the array spacing (resolution) in (z,y,x) order.
    """
    # pydicom is apparently taking into  consideration
    # the endianness (TransferSyntaxUID) when reading the file.
    
    # Sort slices by z-axis position
    dcm_seq.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    array3d = np.stack([dcm2array(dcm) for dcm in dcm_seq])
    
    # Determine pixel spacing
    z = np.float32([dcm.ImagePositionPatient[2] for dcm in dcm_seq])
    
    dz = np.diff(z).mean()
    dy = np.float32([dcm.PixelSpacing[0] for dcm in dcm_seq]).mean()
    dx = np.float32([dcm.PixelSpacing[1] for dcm in dcm_seq]).mean()
    
    spacing = (dz, dy, dx)
    
    return array3d, spacing


# Resample
def resample(array, spacing, new_spacing=(1, 1, 1)):
    """Resamples an array to specified pixel spacing (resolution).
    Args:
        array (numpy.array): input array.
        spacing (tuple): Original pixel spacing in z-y-x order.
        new_spacing (tuple): New pixel spacing in z-y-x order.
    
    Returns:
        Array with new pixel spacing (and thus a new shape).
    """
    
    n = np.float32(array.shape)
    dz = np.float32(spacing)
    dz_ = np.float32(new_spacing)
    
    zoom_factor = (1 - 1/n)*(dz/dz_) + 1/n
    
    return zoom(array, zoom_factor, mode='nearest')


def make_lungs_mask(array, kernel_size=3):
    """ Makes lungs mask from 3D array.
    
        Args:
            array (numpy.array): 3D array of stacked slices.
            kernel_size (int): Size of kernel used in morphological operations (in spacing units).      
        Returns:
            A tuple containing:
                - Sub-array with lung mask.
                - Binarization threshold.
    """

    if array.ndim is not 3:
        raise ValueError('Array must be 3D.')

    kernel2d = np.ones((kernel_size, kernel_size), dtype='uint8')
    kernel3d_zx = np.ones((kernel_size, 1, kernel_size), dtype='uint8')
    kernel3d_y = np.ones((1, kernel_size, 1), dtype='uint8')

    # Find optimal threshold between Air-Lungs (=1) and the rest (water, muscle, fat, bone) (=0)
    thres = filters.threshold_otsu(array.flatten())

    # Binarize
    mask = np.uint8(array < thres)

    def isolate_lungs(mask_slice):
        # Remove noisy pixels
        mask_slice = binary_opening(mask_slice, structure=kernel2d, iterations=2)

        # Pad with ones so the outer air regions get the same label
        mask_slice = np.pad(mask_slice, pad_width=1, mode='constant', constant_values=1)
        labels_slice = measure.label(mask_slice)

        # Kill background, preserve lungs
        background_label = labels_slice[0, 0]
        mask_slice[labels_slice == background_label] = 0

        # Undo padding
        mask_slice = mask_slice[1:-1, 1:-1]

        return mask_slice

    mask_lungs = np.stack([isolate_lungs(x) for x in mask])

    # Dilation along z and x to connect lungs while avoiding the scanner bed
    mask_lungs = binary_dilation(mask_lungs, structure=kernel3d_zx, iterations=4)

    # Keep the biggest volumes
    # Get lungs label (should be the most frequent label besides the background)
    labeled = measure.label(mask_lungs)
    labels, counts = np.unique(labeled, return_counts=True)

    counts = counts[labels > 0]
    labels = labels[labels > 0]
    lungs_label = labels[counts.argmax()]

    mask_lungs = np.uint8(labeled == lungs_label)

    # Finally dilate along the remaining y dimension
    mask_lungs = binary_dilation(mask_lungs, structure=kernel3d_y, iterations=4)

    return mask_lungs, thres


# Extract lung array
def extract_lungs(array, mask, slice_drop_prob=None):
    """ Extracts sub-array containing the lungs.
    
    Args:
        array (numpy.array): 3D array of stacked slices.
        mask (numpy.array): 3D binary array indicating the lung (1) and background voxels (0).
        slice_drop_prob (None or float32): Cumulative volume percentage along z-dimension to drop from volume tails.
    Returns:
        Sub-array with lungs.
    """
    
    # Drop slices from the tails that contain low volume of lungs
    if slice_drop_prob:
        slice_volume = mask.sum(axis=(1, 2))
        total_volume = mask.sum()
        
        idx1 = np.cumsum(slice_volume/total_volume) < slice_drop_prob
        idx2 = np.cumsum(slice_volume/total_volume) > (1 - slice_drop_prob)
        
        kill = np.logical_or(idx1, idx2)
        mask[kill] = 0
    
    # Get bounding box of lung component
    box = bounding_box(mask)
    
    (z1, z2), (y1, y2), (x1, x2) = box
    
    array_lungs = array[z1:z2, y1:y2, x1:x2]
    
    return array_lungs, box


# Get up-right bounding box of array
def bounding_box(array):
    """Computes the up-right bounding box of the
    non-zero elements of an array.
    
    Args:
        array (numpy.array): input array.
    
    Returns:
       List with tuples of array indices corresponding to the limits of
       the smallest bounding box around the nonzero elements of the array.
       [(dim1_min, dim1_max), (dim2_min, dim2_max), ...]
    """
    
    coords = []
    
    for dim in range(array.ndim):
        axes = list(range(0, array.ndim))
        axes.remove(dim)
        
        nonzero = np.any(array, axis=tuple(axes))
        
        dim_min, dim_max = np.where(nonzero)[0][[0, -1]]
        coords.append((dim_min, dim_max))
    
    return tuple(coords)
