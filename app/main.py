from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import shutil

from app.config import TEMP_SEGMENTATION_DIR, TEMP_MRI_DIR  # Updated import
from app.utils.image_processing import load_and_process_images  # Updated import
from app.models.ml_model import MRIClassifier  # Updated import

app = FastAPI()
classifier = MRIClassifier()

@app.post("/predict")
async def predict(
    segmentation_file: UploadFile = File(...),
    mri_files: list[UploadFile] = File(...)
):
    try:
        # Create temporary directories
        os.makedirs(TEMP_SEGMENTATION_DIR, exist_ok=True)
        os.makedirs(TEMP_MRI_DIR, exist_ok=True)

        # Save the segmentation file
        segmentation_path = os.path.join(TEMP_SEGMENTATION_DIR, segmentation_file.filename)
        with open(segmentation_path, "wb") as f:
            shutil.copyfileobj(segmentation_file.file, f)

        # Save the MRI files
        mri_paths = []
        for mri_file in mri_files:
            mri_path = os.path.join(TEMP_MRI_DIR, mri_file.filename)
            with open(mri_path, "wb") as f:
                shutil.copyfileobj(mri_file.file, f)
            mri_paths.append(mri_path)

        # Load and process images
        reference_image, segmentation_resampled = load_and_process_images(
            segmentation_path, mri_paths
        )

        # Extract features and make prediction
        features = classifier.extract_features(reference_image, segmentation_resampled)
        result = classifier.predict(features)

        # Clean up temporary directories
        shutil.rmtree(TEMP_SEGMENTATION_DIR)
        shutil.rmtree(TEMP_MRI_DIR)

        return JSONResponse(content={"prediction": result})

    except Exception as e:
        # Handle errors and clean up temporary directories
        shutil.rmtree(TEMP_SEGMENTATION_DIR, ignore_errors=True)
        shutil.rmtree(TEMP_MRI_DIR, ignore_errors=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)