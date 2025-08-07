# Fuel-Prediction-ML-Model

A Machine Learning Pipeline for Fuel Blend Property Prediction
1. Abstract
This repository contains a comprehensive and robust machine learning pipeline developed in Python for the prediction of ten distinct properties of a fuel blend. The solution leverages advanced feature engineering, a hybrid feature extraction methodology utilizing both Principal Component Analysis (PCA) and a deep learning Autoencoder, and a target-specific modeling strategy. This strategy assigns optimized regression models, including a sophisticated stacking ensemble, to each prediction target. The final model achieves a Mean Absolute Percentage Error (MAPE) of 0.24, demonstrating high predictive accuracy.

2. Pipeline Architecture and Methodology
The predictive pipeline is executed through a series of sequential, modular stages designed to maximize predictive performance.

2.1. Feature Engineering
The initial dataset is augmented through an extensive feature engineering process. New features are systematically generated to capture complex interactions and statistical distributions within the data:

Interaction Features: Product-moment features are created by multiplying the fraction of each of the five components with their corresponding eleven properties.

Statistical Features: For each property, the first four statistical moments (mean, standard deviation, skewness, and kurtosis) are calculated across all five components.

2.2. Feature Extraction
To generate a rich, latent feature space, two parallel feature extraction techniques are employed:

Autoencoder Embeddings: A PyTorch-based deep Autoencoder is trained on the original scaled features to produce 32-dimensional embeddings, capturing non-linear relationships.

Principal Component Analysis (PCA): PCA is applied to the original scaled features to generate 16 orthogonal principal components, providing a dimensionality-reduced, variance-maximized representation of the data.

2.3. Target-Specific Feature Selection
A distinct set of features is selected for each of the ten target properties. This is achieved using Recursive Feature Elimination (RFE) with a LightGBM regressor as the core estimator. For each target, the RFE process selects the 20 most impactful features from the engineered feature set, creating a tailored and optimized feature space for each model.

2.4. Modeling Strategy
A differentiated modeling approach is implemented, assigning a specific, empirically-selected model to each target property. This ensures that the complexity of the model matches the complexity of the prediction task.

Target Property	& Assigned Model/Technique
BlendProperty1, BlendProperty2, BlendProperty4, BlendProperty6 -	Ridge Regression (alpha=1.0)
BlendProperty5	- Random Forest Regressor
BlendProperty10 -	Huber Regressor
BlendProperty3, BlendProperty7, BlendProperty8, BlendProperty9	- Stacking Ensemble

The Stacking Ensemble, reserved for the most complex targets, is architected as follows:

Level 0 - Base Learners:

LightGBM Regressor

CatBoost Regressor

Random Forest Regressor

A custom 4-layer Deep Neural Network implemented in PyTorch

Level 1 - Meta-Learner:

XGBoost Regressor, which trains on the predictions of the base learners to produce the final output.

2.5. Validation
All model training and evaluation are performed within a robust 5-Fold Cross-Validation framework. This ensures that the reported performance is generalizable and mitigates the risk of overfitting. The final predictions for the test dataset are an average of the predictions from each of the five folds.

3. Technology Stack
Data Manipulation: Pandas, NumPy

Scientific Computing: SciPy

Machine Learning: Scikit-learn (for Preprocessing, RFE, and models)

Deep Learning: PyTorch

Gradient Boosting: LightGBM, CatBoost, XGBoost

4. Installation and Execution
4.1. Prerequisites
Python 3.8 or newer.

For hardware acceleration, a CUDA-enabled GPU is recommended. The script is configured to automatically utilize the GPU for PyTorch, CatBoost, LightGBM, and XGBoost if available, and will revert to CPU-based execution otherwise.

4.2. Setup Instructions
Clone the repository to a local machine:

git clone https://github.com/your-username/fuel-property-prediction.git
cd fuel-property-prediction
Install the required dependencies using the provided requirements.txt file:
pip install -r requirements.txt

4.3. Data Configuration
The script is configured to read train.csv and test.csv from a directory path specified on lines 36-37. Please ensure the data files are located at this path or update the path in the script accordingly.

4.4. Execution
Execute the main script from the command line:
python Fuel_Prediction.py
