import SimpleITK as sitk

def resample_segmentation_to_image(segmentation, reference_image):
    """
    Resample the segmentation image to match the reference image's size, spacing, and direction.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing(reference_image.GetSpacing())
    resampler.SetSize(reference_image.GetSize())
    resampler.SetOutputDirection(reference_image.GetDirection())
    resampler.SetOutputOrigin(reference_image.GetOrigin())
    return resampler.Execute(segmentation)

def load_and_process_images(segmentation_path, mri_paths):
    """
    Load and process the segmentation and MRI images.
    """
    # Load the segmentation image
    segmentation_image = sitk.ReadImage(segmentation_path)

    # Load the MRI images as a single volume
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(mri_paths)
    reference_image = reader.Execute()

    # Resample the segmentation to match the MRI volume
    segmentation_resampled = resample_segmentation_to_image(segmentation_image, reference_image)

    return reference_image, segmentation_resampled