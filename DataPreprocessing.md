# Data preprocessing

## Exploratory data analysis

### DICOM format

1. Padding
3. Transform raw values to Hounsfield Units (HU).
3. Stack slices into a 3D array
2. Isotropic resampling
3. Extraction of lung area


| Field                   | Meaning                                                                     | Example value |
|-------------------------|-----------------------------------------------------------------------------|---------------|
| ImageOrientationPatient | Orientation of image plane respect to the patient’s axes of coordinates     | [1,0,0,0,1,0] |
| ImagePositionPatient    | Position of image plane respect to the equipment’s axes of coordinates      | -50 mm        |
| PixelPaddingValue       | Value of pixels outside of scanner area for padding to a rectangular format | 2000          |
| PixelSpacing            | Gap between pixels (spatial resolution)                                     | 0.65 mm       |
| RescaleIntercept (Ri)   | Intercept of linear transformation Raw -> HU                                | -1024         |
| RescaleSlope (Rs)       | Slope of linear transformation Raw -> HU                                    | 1             |
| Rows                    | Number of rows in slice                                                     | 512           |
| Columns                 | Number of columns in slice                                                  | 512           |
| BitsStored              | Bit resolution of pixels in slice                                           | 12            |
| SamplesPerPixel         | Number of channels per pixel                                                | 1             |



|-------------------------------|-------------------------|-----------------------|
| **PixelData**                 | **RescaleSlope**        | CompressionCode       |
| **BitsStored**                | **Rows**                | ImageDimensions       |
| BurnedInAnnotation            | **Columns**             | ImageFormat           |
| **ImageOrientationPatient**   | **SamplesPerPixel**     | ImageLocation         |
| **ImagePositionPatient**      | SeriesDescription       | NumberOfFrames        |
| Modality                      | SeriesInstanceUID       | PixelAspectRatio      |
| ~~PatientBirthDate~~          | SeriesNumber            | LossyImageCompression |
| ~~PatientName~~               | SliceLocation           | Laterality            |
| **PhotometricInterpretation** | PatientOrientation      | SourceImageSequence   |
| PixelPaddingValue             | PlanarConfiguration     | TemporalPositionIndex |
| **PixelRepresentation**       | RescaleType             | VolumetricProperties  |
| **PixelSpacing**              | LargestImagePixelValue  | ~~Allergies~~         |
| **RescaleIntercept**          | SmallestImagePixelValue | ~~PregnancyStatus~~   |
