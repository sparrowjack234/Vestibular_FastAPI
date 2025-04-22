# Vestibular_FastAPI
•⁠  ⁠Developed ML models to predict tumor response post-Cyberknife®️ radiosurgery using radiomic features from pre-treatment MRI.
•⁠  ⁠Extracted 851 features using PyRadiomics; performed preprocessing (bias correction, normalization) and 3D segmentation with 3D Slicer.
•⁠  ⁠Trained and evaluated classifiers (Neural Network, SVM, XGBoost, Random Forest) using nested cross-validation and LASSO-based feature selection.
•⁠  ⁠Achieved 73% balanced accuracy at 24 months with Neural Network; handled class imbalance using SMOTE.
•⁠  ⁠*Built a FastAPI-based web service* to deploy the prediction pipeline, enabling MRI file upload and automated response prediction.
