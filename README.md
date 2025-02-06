# Earthquake Magnitude Prediction using Random Forest

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Source & Description](#data-source--description)
3. [Problem Definition](#problem-definition)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Training](#model-training)
6. [Evaluation](#evaluation)
7. [Model Deployment](#model-deployment)
8. [Running the Project](#running-the-project)
9. [Dependencies](#dependencies)
10. [Contributing](#contributing)
11. [License](#license)

## Project Overview
This project aims to predict the **magnitude of earthquakes** based on several seismic features such as **depth, latitude, longitude**, and **number of stations** reporting the earthquake. A **Random Forest** regressor model is used to train and predict the earthquake magnitude. The project includes a **Streamlit** UI for user interaction and a **FastAPI** deployment for model serving.

## Data Source & Description
- **Dataset Source**: [Kaggle Earthquake Data](https://www.kaggle.com/datasets/)
- **Features**:
  - **`depth`**: Depth of the earthquake (in km).
  - **`latitude`**: Latitude of the earthquake.
  - **`longitude`**: Longitude of the earthquake.
  - **`nst`**: Number of stations reporting the earthquake.
  - **`mag`**: Earthquake magnitude (target variable).

## Problem Definition
The goal is to predict the magnitude of earthquakes based on seismic data. We chose **Random Forest Regressor** as the model to solve this regression problem.

## Data Preprocessing
- Missing values are handled appropriately.
- Numerical features are scaled using **StandardScaler**.
- Categorical features are encoded (if any).
- The data is split into **80% training** and **20% testing** sets.

## Model Training
- A **Random Forest Regressor** is trained on the preprocessed dataset.
- **Mean Squared Error (MSE)** and **R² Score** are used for model evaluation.

## Evaluation
The trained model is evaluated using:
- **MSE**: Measures prediction accuracy.
- **R² Score**: Indicates the proportion of variance explained by the model.

## Model Deployment
The trained model is deployed as an API using **FastAPI** and can be accessed programmatically. A **Streamlit** app is also created to allow users to input earthquake parameters and get real-time magnitude predictions.

## Running the Project
To run this project, follow the steps below:

### Setup the Environment:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/earthquake-prediction.git
   cd earthquake-prediction
