# 🎬 MovieIQ – Predictive Analytics on Movie Success

MovieIQ is a machine learning project that predicts whether a movie is likely to be successful based on key movie attributes such as budget, popularity, runtime, and vote average. The project combines data analysis, statistical testing, predictive modeling, and an interactive Streamlit dashboard.

---

## 📌 Project Overview

The objective of this project is to analyze movie data and build a predictive model that classifies movies as **Successful** or **Unsuccessful**.

The project includes:

- Data Cleaning & Preparation
- Exploratory Data Analysis (EDA)
- Statistical Testing
- Random Forest Classification
- Interactive Streamlit Dashboard
- Movie Success Prediction

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Joblib

---

## Project Structure

```
MovieIQ/
│
├── assets/
│   ├── budget_vs_revenue.png
│   ├── genre_frequency.png
│   ├── genre_success_rate.png
│   ├── popularity_vs_success.png
│   ├── runtime_vs_success.png
│   ├── vote_average_vs_success.png
│   ├── correlation_heatmap.png
│   ├── success_distribution.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── models/
│   └── rf_model.pkl
│
├── clean_movies.csv
├── MovieIQ.py
├── requirements.txt
└── README.md
```

---

## Exploratory Data Analysis

The project includes the following visualizations:

- Budget vs Revenue
- Genre Frequency
- Genre Success Rate
- Popularity vs Success
- Runtime vs Success
- Vote Average vs Success
- Correlation Heatmap
- Success Distribution
- Confusion Matrix
- Feature Importance

---

## Machine Learning Model

A **Random Forest Classifier** is trained using the following features:

- Budget
- Popularity
- Runtime
- Vote Average

Target Variable:

- Success (1)
- Failure (0)

Model evaluation includes:

- Accuracy
- Confusion Matrix
- Classification Report
- Feature Importance

---

## Features of the Dashboard

- Dataset Overview
- Interactive EDA Visualizations
- Movie Success Prediction
- Model Confidence Score
- User-Friendly Interface

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/MovieIQ.git
```

Navigate to the project folder:

```bash
cd MovieIQ
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run MovieIQ.py
```

---

## Dashboard Preview

Add screenshots of your Streamlit dashboard here after running the application.

Example:

- Home Page
- EDA Dashboard
- Prediction Page

---


## 👨‍💻 Author

**Anamey Deorukhkar**

BBA (Data Analytics) Student

---

## 📄 License

This project is developed for educational and learning purposes.
