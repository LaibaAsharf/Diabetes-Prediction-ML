# Diabetes Prediction Using Machine Learning

## Project Overview
This project predicts the likelihood of diabetes using Machine Learning. It uses a Logistic Regression model trained on a diabetes dataset and provides an interactive interface built with Streamlit.

## Features
- Diabetes prediction based on health-related input features.
- Data preprocessing and feature scaling.
- Logistic Regression classification model.
- Interactive Streamlit web application.
- Model evaluation using accuracy and confusion matrix.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Logistic Regression
- StandardScaler
- Streamlit

## Project Structure
- `app.py` – Streamlit application.
- `main.py` – Main project script.
- `model.py` – Model training and prediction logic.
- `model.pkl` – Saved trained model.
- `scaler.pkl` – Saved feature scaler.
- `Dataset/diabetes.csv` – Dataset.

## How to Run
1. Clone the repository.
2. Install the required libraries:
   ```bash
   pip install pandas scikit-learn streamlit
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```
## Screenshots

### Application Interface
![Diabetes Prediction App](Screenshot%202026-10-11%20161212.png)

### Prediction Result
![Prediction Output](Screenshot%202026-10-11%20161407.png)
## Disclaimer
This project is developed for educational purposes only. Its predictions are not a substitute for professional medical advice.
