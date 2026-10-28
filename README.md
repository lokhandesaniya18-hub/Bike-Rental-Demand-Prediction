# 🚲 Bike Rental Demand Prediction

## 📌 Project Overview
This project uses **Supervised Machine Learning – Regression** to predict daily bike rental demand based on weather conditions, seasonal factors, and calendar information.

The goal is to understand rental patterns and build a model that estimates the total number of bike rentals.

## 🎯 Objectives
- Perform data cleaning and Exploratory Data Analysis (EDA).
- Analyze the impact of weather, seasons, and time on rental demand.
- Build and compare multiple regression models.
- Evaluate models using MAE, RMSE, and R² score.
- Select the best-performing model.

## 📂 Dataset
- `day.csv` – Daily bike rental data.
- `hour.csv` – Hourly bike rental data.

**Target Variable:** `cnt` — total number of bike rentals.

## 🛠️ Technologies Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib, Jupyter Notebook.

## 🤖 Machine Learning
This is a regression problem because the target variable `cnt` is numerical.

Regression models evaluated include Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting, SVR, and KNN.

The **Tuned Gradient Boosting Regressor** was selected based on its test performance.

| Metric | Score |
|---|---:|
| MAE | 874.55 |
| RMSE | 1041.33 |
| R² Score | 0.6914 |

## 🚀 How to Run

1. Clone or download this repository.
2. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
   ```

3. Open Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Open `bike_rental.ipynb` and run the cells in order.

**Project Files:**
- `bike_rental.ipynb` – Project notebook
- `day.csv` – Daily bike rental dataset
- `hour.csv` – Hourly bike rental dataset
- `bike_rental_model.pkl` – Saved machine learning model

Keep the notebook, datasets, and model file in the appropriate project folder.

## ✅ Conclusion
The project demonstrates data analysis, feature selection, regression model comparison, and model evaluation for bike rental demand prediction.


