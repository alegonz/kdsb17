"""
##### Data generator

* random rotation (48 possibilities)
* zero-mean
* scaling
* add random offset value to all pixels
"""

import os

from glob import glob
import dicom
import numpy as np
from skimage import filters, measure
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import binary_erosion, binary_dilation
from matplotlib import pyplot as plt

# Read sequence of dicom files of a patient
def read_dcm_sequence(patientid, base_path):
    """Read dicom files in path of a patient.
    
    Args:
        patientid (str): ID of the patient (patient foldername).
        base_path (str): Path containing the patient folders.
    
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
    
    # Check basic requirements
    rescale_type_raised = False
    series_description_raised = False
    for dcm in dcm_seq:
        
        # These are Type 1 elements
        assert dcm.Modality == 'CT', 'Expected Modality=CT, got %s.' % dcm.Modality
        assert dcm.BitsAllocated == 16, 'Expected BitsAllocated=16, got %d.' % dcm.BitsAllocated
        assert dcm.PhotometricInterpretation == 'MONOCHROME2', 'Expected PhotometricInterpretation=MONOCHROME2, got %s.' % dcm.PhotometricInterpretation
        assert dcm.Rows == 512, 'Expected Rows=512, got %d.' % dcm.Rows
        assert dcm.Columns == 512, 'Expected Columns=512, got %d.' % dcm.Columns
        assert dcm.SamplesPerPixel == 1, 'Expected SamplesPerPixel=1, got %d.' % dcm.SamplesPerPixel
        assert dcm.ImageOrientationPatient == [1,0,0,0,1,0], 'Expected ImageOrientationPatient=[1,0,0,0,1,0], got %s.' % dcm.ImageOrientationPatient
        
        if dcm.BitsStored == 12:
            assert dcm.pixel_array.max() < 4096, 'Pixel array values exceed 4095, but BitsStored=12.'
        
        try:
            assert dcm.RescaleType in ['HU','US'], 'Expected RescaleType={empty,HU,US}, got %s.' % dcm.RescaleType
        except:
            if not rescale_type_raised:
                rescale_type_raised = True
                print(dcm.PatientID, 'RescaleType unavailable.')
            
        try:
            assert dcm.SeriesDescription == 'Axial', 'Expected SeriesDescription=Axial, got %s.' % dcm.SeriesDescription
        except:
            if not series_description_raised:
                series_description_raised = True
                print(dcm.PatientID, 'SeriesDescription unavailable.')
    
    # Check if overlapping acquisitions exist
    # Edge case with two overlapping acquisitions: b8bb02d229361a623a4dc57aa0e5c485
    positions = {}
    for dcm in dcm_seq:
        try:
            acq = dcm.AcquisitionNumber
        except:
            acq = None
        
        if acq in positions:
            positions[acq].append(float(dcm.ImagePositionPatient[2]))
        else:
            positions[acq] = [float(dcm.ImagePositionPatient[2])]
    
    if len(positions) > 1:
        intervals = [(acq,min(p),max(p),len(p)) for acq, p in positions.items()]
        intervals.sort(key=lambda interval: interval[1])
        
        (_, _, Mp, _) = intervals[0]
        for (acq, m, M, _) in intervals[1:]:
            if m <= Mp:
                print(dcm.PatientID, 'Found overlap at acquisition', acq)
                
                intervals.sort(key=lambda interval: interval[3])
                acq = intervals[-1][0]
                
                print(dcm.PatientID, 'Keeping acquisition', acq)
                
                dcm_seq = filter(lambda dcm: dcm.AcquisitionNumber == acq, dcm_seq)
                
                break
            
            else:
                Mp = M
    
    # Sort slices by z-axis position
    dcm_seq.sort(key=lambda dcm: float(dcm.ImagePositionPatient[2]))
    
    # Check if pixel spacing if uniform across slices.
    z = np.float32([dcm.ImagePositionPatient[2] for dcm in dcm_seq])
    
    dz_std = np.diff(z).std()
    dy_std = np.float32([dcm.PixelSpacing[0] for dcm in dcm_seq]).std()
    dx_std = np.float32([dcm.PixelSpacing[1] for dcm in dcm_seq]).std()
    
    if (dz_std > tol) or (dy_std > tol) or (dx_std > tol):
        raise Exception('Pixel spacing is not uniform.')


# Convert raw dicom pixel values to array in Hounsfield Units.
def dcm2array(dcm):
    """Converts the raw pixel array in dicom data to an image array in HU units.
    Args:
        dcm (dicom.dataset.FileDataset): input dicom data.
    
    Returns:
        Image array in HU units.
    """
    
    # These are Type 1 so they shouldn't raise an error
    slope = dcm.RescaleSlope
    intercept = dcm.RescaleIntercept
    pixel_representation = dcm.PixelRepresentation
    
    # This is Type 2, so it might not be present in the dicom file
    try:
        pixel_padding_value = dcm.PixelPaddingValue
    except:
        pixel_padding_value = None
    
    # Apparently pydicom is already taking into consideration
    # the PixelRepresentation and BitsAllocated when making the pixel_array.
    # However is not considering BitsStored nor HighBit.
    array = dcm.pixel_array
    
    if isinstance(pixel_padding_value,bytes):
        pixel_padding_value = int.from_bytes(
            pixel_padding_value, byteorder='little', signed=(pixel_representation==1))
    
    # Set padded area to air (HU=-1000)
    if pixel_padding_value:
        array[array == pixel_padding_value] = -1000 - intercept
    
    # Transform to Hounsfield Units.
    array = np.float64(array)
    array *= slope
    array += intercept
    array = np.int16(array)
    
    # DEPRECATED:
    # * Misunderstanding of pixel sample storaging.
    # * PixelRepresentation just refers to the raw pixel sample sign.
    # * Furthermore, nowadays overlay data is stored separately.
    #
    # Check range of array values; CT scans are at most 12 bits.
    # top = 4095 if pixel_representation == 0 else 2047
    # bottom = 0 if pixel_representation == 0 else -2048
    # assert bottom <= array.max() <= top, '(dcm2array) array values outside [%d,%d] interval' % (bottom,top)
    
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
    dcm_seq.sort(key=lambda dcm: float(dcm.ImagePositionPatient[2]))
    
    array3d = np.stack([dcm2array(dcm) for dcm in dcm_seq])
    
    # Determine pixel spacing
    z = np.float32([dcm.ImagePositionPatient[2] for dcm in dcm_seq])
    
    dz = np.diff(z).mean()
    dy = np.float32([dcm.PixelSpacing[0] for dcm in dcm_seq]).mean()
    dx = np.float32([dcm.PixelSpacing[1] for dcm in dcm_seq]).mean()
    
    spacing = [dz,dy,dx]
    
    return array3d, spacing


# Resample
def resample(array, spacing, new_spacing=[1,1,1]):
    """Resamples an array to specified pixel spacing (resolution).
    Args:
        array (numpy.array): input array.
        spacing (list): Original pixel spacing in z-y-x order.
        new_spacing (list): New pixel spacing in z-y-x order.
    
    Returns:
        Array with new pixel spacing (and thus a new shape).
    """
    
    N = np.float32(array.shape)
    dz = np.float32(spacing)
    dz_ = np.float32(new_spacing)
    
    zoomfactor = (1 - 1/N)*(dz/dz_) + 1/N
    
    return zoom(array, zoomfactor, mode='nearest')



# Extract lung array
def extract_lungs(array, kernel_size=3, slice_drop_prob=None):
    """ Extracts sub-array containing the lungs.
    
    Args:
        array (numpy.array): 3D array of stacked slices.
        kernel_size (int): Size of erosion/dilation kernel for segmentation (in spacing units).
    
    Returns:
        A tuple containing:
            - Sub-array with lungs.
            - Binarization threshold.
            - Bounding box coordinates.
    """
    
    if array.ndim is not 3:
        raise ValueError('Array must be 3D.')
    
    # Find optimal threshold between Air-Lungs (=1) and the rest (water, muscle, fat, bone) (=0)
    thres = filters.threshold_otsu(array.flatten())
    mask = array < thres
    
    # Pad with ones so the outer air regions get the same label
    mask = np.pad(mask, pad_width=((0,0),(1,1),(1,1)), mode='constant', constant_values=1)
    mask_labeled = measure.label(mask)
    
    # Kill background, preserve lungs
    background_label = mask_labeled[0,0,0]
    mask[mask_labeled == background_label] = 0
    
    # Undo padding
    mask = mask[1:-1,1:-1,1:-1]
    
    # Erode noisy voxels and dilate to fill lungs and add some margin around
    kernel = np.ones([kernel_size]*3, dtype='bool')
    
    mask_lungs = binary_erosion(mask, structure=kernel, iterations=1)
    mask_lungs = binary_dilation(mask_lungs, structure=kernel, iterations=2)
    
    # Drop slices from the tails that contain low volume of lungs
    if slice_drop_prob:
        slice_volume = mask_lungs.sum(axis=(1,2))
        total_volume = mask_lungs.sum()
        
        idx1 = np.cumsum(slice_volume/total_volume) < slice_drop_prob/2
        idx2 = np.cumsum(slice_volume/total_volume) > (1 - slice_drop_prob/2)
        
        kill = np.logical_or(idx1, idx2)
        mask_lungs[kill] = 0
    
    # Get bounding box of lung component
    box = bounding_box(mask_lungs)
    
    (z1,z2), (y1,y2), (x1,x2) = box
    
    array_lungs = array[z1:z2,y1:y2,x1:x2]
    
    return array_lungs, thres, box


# Extract lung array (old method)
def extract_lungs_old(array, erosion_size=3, dilation_size=8, drop_rate=0.50):
    """ Extracts sub-array containing the lungs.
    
    Args:
        array (numpy.array): Array of stacked slices.
        erosion_size (int): Size of erosion kernel for segmentation (units: mm).
        dilation_size (int): Size of dilation kernel for segmentation (units: mm).
        drop_rate (float): A ratio when sliced from the center. Must be in [0,1].
    
    Returns:
        A tuple containing:
            - Sub-array with lungs.
            - Binarization threshold.
            - Bounding box coordinates.
    """
    
    if not 0 <= drop_rate <= 1:
        raise ValueError('Invalid drop_rate. Must be a value in [0,1].')
    
    # Find optimal threshold between Air-Lungs and the rest (water, muscle, fat, bone)
    thres = filters.threshold_otsu(array.flatten())
    mask = array < thres
    
    # Erode noisy voxels and dilate to fill lungs
    kernel_erosion = np.ones([erosion_size]*array.ndim, dtype='bool')
    kernel_dilation = np.ones([dilation_size]*array.ndim, dtype='bool')
    
    mask_eroded = binary_erosion(mask, structure=kernel_erosion, iterations=1)
    mask_dilated = binary_dilation(mask_eroded, structure=kernel_dilation, iterations=1)
    
    # Label connected parts
    mask_labeled = measure.label(mask_dilated)
    
    # Determine the most common label with a representative central volume
    bottom = drop_rate/2
    top = 1-drop_rate/2
    (z1,z2), (y1,y2), (x1,x2) = [(int(bottom*n), int(top*n)) for n in mask_labeled.shape]
    
    label, counts = np.unique(mask_labeled[z1:z2,y1:y2,x1:x2], return_counts=True)
    lungs_label = label[counts.argmax()]
    
    # Get bounding box of lung component
    box = bounding_box(mask_labeled==lungs_label)
    
    (z1,z2), (y1,y2), (x1,x2) = box
    
    array_lungs = array[z1:z2,y1:y2,x1:x2]
    
    return array_lungs, thres, box


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
        axes = list(range(0,array.ndim))
        axes.remove(dim)
        
        nonzero = np.any(array, axis=tuple(axes))
        
        dim_min, dim_max = np.where(nonzero)[0][[0, -1]]
        coords.append((dim_min, dim_max))
    
    return coords

