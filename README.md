# 🌍 Earthquake Magnitude Prediction using Random Forest

## Project Overview
This project aims to predict the **magnitude of earthquakes** using historical earthquake data. The prediction model is built using a **Random Forest Regressor** and is deployed through a **FastAPI** server for programmatic access. A user-friendly **Streamlit** app is also created for direct interaction with the model.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Source & Description](#data-source--description)
3. [Problem Definition](#problem-definition)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Training](#model-training)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Model Deployment](#model-deployment)
8. [Running the Project](#running-the-project)
9. [Dependencies](#dependencies)
10. [Contributing](#contributing)
11. [License](#license)

## Data Source & Description
The dataset for this project comes from publicly available earthquake data, such as data from the **United States Geological Survey (USGS)** or **Kaggle**.
- **File Format**: CSV
- **Key Features**:
  - `depth` (float): Depth of the earthquake in kilometers.
  - `latitude` (float): Latitude of the earthquake.
  - `longitude` (float): Longitude of the earthquake.
  - `nst` (int): Number of reporting stations.
  - `mag` (float): Earthquake magnitude (target variable).

## Problem Definition
We aim to predict the magnitude of an earthquake based on seismic features like depth, latitude, longitude, and the number of stations reporting the earthquake. This is a regression problem, and we use the **Random Forest Regressor** model to solve it.

## Data Preprocessing
1. **Handling Missing Values**: Imputing or removing missing data.
2. **Feature Scaling**: Standardizing numerical features using **StandardScaler**.
3. **Outlier Detection**: Identifying and handling outliers to improve model performance.

## Model Training
The model used is the **Random Forest Regressor**, which is trained on historical earthquake data to predict earthquake magnitudes. **Hyperparameter tuning** is performed to improve performance.

## Evaluation Metrics
The model is evaluated using:
- **Mean Squared Error (MSE)**: Measures the prediction accuracy.
- **R² Score**: Indicates how well the model explains the variance in the data.

## Model Deployment
The trained model is deployed using **FastAPI**, and predictions can be made programmatically. A **Streamlit** app is provided to allow users to input earthquake parameters and receive magnitude predictions.

## Running the Project

### 1. Setup the Environment:
Clone the repository:
```bash
git clone https://github.com/yourusername/earthquake-prediction.git
cd earthquake-prediction
```
## Create and activate the virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```
## Install dependencies:
```bash
pip install -r requirements.txt
```
## 2. Running the Streamlit App:
```bash
Start the Streamlit app:
```
streamlit run app/app.py

The app will open in your browser where you can interact with the model.

## 3. Running the FastAPI Server:
Run the FastAPI server:
```bash
uvicorn deployment.api:app --reload
The FastAPI server will be available at http://localhost:8000.
```
## Dependencies
-`Python 3.x`
-`pandas`
-`scikit-learn`
-`numpy`
-`matplotlib`
-`seaborn`
-`fastapi`
-`streamlit`
-`uvicorn`

## Contributing
Feel free to fork this repository, open issues, or create pull requests to improve this project.

## License
This project is licensed under the MIT License - see the LICENSE file for details.