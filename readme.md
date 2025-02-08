# Election Prediction Project

## Project Overview
This project predicts election outcomes using machine learning techniques. It utilizes historical election data and various demographic features to train a predictive model.

## Problem Definition
The goal is to build a model that can accurately predict election results based on input features such as voter demographics, past election trends, and other relevant factors.

## Data Source and Description
- **Dataset:** The data is loaded from `election.csv`.
- **Preprocessing Steps:**
  - Handling missing values by dropping NaNs.
  - Encoding categorical variables using one-hot encoding.
  - Splitting the dataset into training and testing sets.

## Exploratory Data Analysis (EDA)
- **Visualizations Used:**
  - Histograms for feature distributions.
  - Correlation heatmaps to understand relationships between variables.
  - Confusion matrices to evaluate model performance.

## Model Selection and Training
- **Algorithm Used:** RandomForestClassifier.
- **Training Process:**
  - The dataset is split into training and test sets.
  - A RandomForestClassifier is trained on the processed data.
  - The model is evaluated using accuracy, confusion matrix, and classification report.

## Model Evaluation
- **Metrics Used:**
  - Accuracy Score
  - Confusion Matrix
  - Classification Report
- **Results:**
  - The model's accuracy and performance on the test set are assessed using the above metrics.

## Deployment
- **Deployment Strategy:**
  - The model is saved using Joblib (`model.pkl`).
  - A FastAPI-based API is used to make predictions.
- **How to Run the API:**
  - Load the trained model.
  - Define an API endpoint to receive input data.
  - Return predictions based on model inference.

## How to Reproduce the Project
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Load the dataset:
   ```python
   data = pd.read_csv('election.csv')
   ```
3. Preprocess the data:
   ```python
   data = data.dropna()
   data = pd.get_dummies(data, drop_first=True)
   ```
4. Train the model:
   ```python
   model = RandomForestClassifier()
   model.fit(X_train, y_train)
   ```
5. Save and load the model:
   ```python
   joblib.dump(model, 'model.pkl')
   model = joblib.load('model.pkl')
   ```
6. Run the FastAPI application:
   ```sh
   uvicorn app:main --reload
   ```


