import joblib
import numpy as np
from radiomics import featureextractor
from app.config import MODEL_PATH, FEATURE_NAMES_PATH  # Updated import

class MRIClassifier:
    def __init__(self):
        # Load the pre-trained Random Forest model
        self.model = joblib.load(MODEL_PATH)
        
        # Load selected feature names
        with open(FEATURE_NAMES_PATH, "r") as f:
            self.selected_feature_names = [line.strip() for line in f.readlines()]
        
        # Initialize the radiomics feature extractor
        self.feature_extractor = featureextractor.RadiomicsFeatureExtractor()

    def extract_features(self, reference_image, segmentation_resampled):
        """
        Extract radiomics features from the images.
        """
        features = self.feature_extractor.execute(reference_image, segmentation_resampled)
        
        # Select only the features specified in the .txt file
        extracted_features = {
            key: features[key] 
            for key in self.selected_feature_names 
            if key in features
        }
        
        return extracted_features

    def predict(self, features):
        """
        Make a prediction using the extracted features.
        """
        model_input_features = np.array(list(features.values())).reshape(1, -1)
        prediction = self.model.predict(model_input_features)
        
        return "Increase" if prediction[0] == 1 else "Not Increase"