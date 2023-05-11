# Heart Disease Prediction

## Dataset Information

This dataset contains information about patients and their heart health. The data was sourced from [Kaggle](https://www.kaggle.com/johnsmith88/heart-disease-dataset).

The dataset includes the following features:

- Age: age of the patient [years]
- Sex: sex of the patient [M: Male, F: Female]
- ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
- RestingBP: resting blood pressure [mm Hg]
- Cholesterol: serum cholesterol [mm/dl]
- FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
- RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
- MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
- ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
- Oldpeak: oldpeak = ST [Numeric value measured in depression]
- ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
- HeartDisease: output class [1: heart disease, 0: Normal]

## Research Environment

The research environment included the following steps:

1. EDA (distribution checking, checking outliers, missing values)
2. Feature engineering (data transformation by scaling, encoding categorical feature, outliers handling using IQR)
3. Model training (KNN)
4. Hyperparameter training

## Production Environment

The production environment included the following files:

1. `requirements.txt`: A setup file for installing required libraries
2. `exceptions.py`: A custom exception handling file
3. `logger.py`: A logger file for logging
4. `data_ingestion.py`: A file for defining train and test data path and returning train/test path
5. `data_transformation.py`: A file for reading data from data ingestion path and transforming data, like scaling numerical features, one-hot encoding categorical features, and fitting these transformations on train/test data and saving the preprocessor.pkl file and getting train/test array.
6. `model_trainer.py`: A file for reading train and test array from data transformation, splitting the data `X_train`, `y_train`, `X_test`, `y_test`, defining a dictionary of models to check which model performs best, some models include:

- Random Forest
- Decision Tree
- Gradient Boosting
- K-Neighbours
- XGBClassifier
- CatBoost
- AdaBoost

The file also performs hyperparameter tuning by defining a dictionary of params for the respective models and saves the model.pkl file which performs >60 r2_score.
7. `utils.py`: A file containing all the helper functions like `save_object()` to save pkl file and `evaluate_model()` to evaluate each model one by one.
8. The `notebook` folder contains the research environment file and heart.csv file.
9. The `artifacts` folder contains the trained, test, and raw data.

## Installation and Usage

1. Clone the repository: `git clone https://github.com/your_username/heart-disease-prediction.git`
2. Install the required libraries: `pip install -r requirements.txt
