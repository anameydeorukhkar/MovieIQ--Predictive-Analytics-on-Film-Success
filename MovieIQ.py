#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================
# MovieIQ - Streamlit Dashboard
# ============================

import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import os

# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="MovieIQ",
    page_icon="🎬",
    layout="wide"
)

# ----------------------------
# Load Dataset and Model
# ----------------------------

movies = pd.read_csv("clean_movies.csv")

model = joblib.load("models/rf_model.pkl")

# ----------------------------
# Title
# ----------------------------

st.title("🎬 MovieIQ")
st.subheader("Predictive Analytics on Movie Success")

st.markdown("---")

# ----------------------------
# Sidebar Navigation
# ----------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a Page",
    (
        "Home",
        "Exploratory Data Analysis",
        "Movie Success Prediction"
    )
)

# ==========================================================
# HOME PAGE
# ==========================================================

if page == "Home":

    st.header("Dataset Overview")

    st.write(
        """
        MovieIQ is a machine learning application that predicts
        whether a movie is likely to be successful based on
        important movie characteristics.

        This project includes:

        • Data Preparation
        • Exploratory Data Analysis
        • Statistical Testing
        • Random Forest Classification
        • Movie Success Prediction
        """
    )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(movies.head())

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Number of Movies",
            movies.shape[0]
        )

    with col2:

        st.metric(
            "Number of Features",
            movies.shape[1]
        )

    st.markdown("---")

    st.subheader("Dataset Columns")

    st.write(list(movies.columns))

    st.markdown("---")

    st.subheader("Statistical Summary")

    st.dataframe(
        movies.describe()
    )

    st.markdown("---")

    st.success("Dataset Loaded Successfully!")

# ==========================================================
# EDA PAGE
# ==========================================================

elif page == "Exploratory Data Analysis":

    st.header("Exploratory Data Analysis")

    st.write(
        """
        This section displays all visualizations created during
        the Exploratory Data Analysis phase.
        """
    )

    st.markdown("---")

    # ----------------------------
    # Budget vs Revenue
    # ----------------------------

    st.subheader("1. Budget vs Revenue")

    st.image(
        "assets/budget_vs_revenue.png",
        use_container_width=True
    )

    st.info(
        "This scatter plot shows the relationship between movie budget and revenue."
    )

    st.markdown("---")

    # ----------------------------
    # Genre Frequency
    # ----------------------------

    st.subheader("2. Most Common Movie Genres")

    st.image(
        "assets/genre_frequency.png",
        use_container_width=True
    )

    st.info(
        "Displays the number of movies belonging to each genre."
    )

    st.markdown("---")

    # ----------------------------
    # Genre Success Rate
    # ----------------------------

    st.subheader("3. Success Rate by Genre")

    st.image(
        "assets/genre_success_rate.png",
        use_container_width=True
    )

    st.info(
        "Shows the average success rate for each movie genre."
    )

    st.markdown("---")

    # ----------------------------
    # Popularity vs Success
    # ----------------------------

    st.subheader("4. Popularity vs Movie Success")

    st.image(
        "assets/popularity_vs_success.png",
        use_container_width=True
    )

    st.info(
        "Compares popularity scores of successful and unsuccessful movies."
    )

    st.markdown("---")

    # ----------------------------
    # Runtime vs Success
    # ----------------------------

    st.subheader("5. Runtime vs Movie Success")

    st.image(
        "assets/runtime_vs_success.png",
        use_container_width=True
    )

    st.info(
        "Shows how runtime varies between successful and unsuccessful movies."
    )

    st.markdown("---")

    # ----------------------------
    # Vote Average vs Success
    # ----------------------------

    st.subheader("6. Vote Average vs Movie Success")

    st.image(
        "assets/vote_average_vs_success.png",
        use_container_width=True
    )

    st.info(
        "Compares audience ratings between successful and unsuccessful movies."
    )

    st.markdown("---")

    # ----------------------------
    # Correlation Heatmap
    # ----------------------------

    st.subheader("7. Correlation Heatmap")

    st.image(
        "assets/correlation_heatmap.png",
        use_container_width=True
    )

    st.info(
        "Displays the correlation among all numerical variables."
    )

    st.markdown("---")

    # ----------------------------
    # Success Distribution
    # ----------------------------

    st.subheader("8. Distribution of Successful Movies")

    st.image(
        "assets/success_distribution.png",
        use_container_width=True
    )

    st.info(
        "Shows the number of successful and unsuccessful movies."
    )

    st.markdown("---")

    # ----------------------------
    # Confusion Matrix
    # ----------------------------

    st.subheader("9. Confusion Matrix")

    st.image(
        "assets/confusion_matrix.png",
        use_container_width=True
    )

    st.info(
        "Shows the classification performance of the Random Forest model."
    )

    st.markdown("---")

    # ----------------------------
    # Feature Importance
    # ----------------------------

    st.subheader("10. Feature Importance")

    st.image(
        "assets/feature_importance.png",
        use_container_width=True
    )

    st.info(
        "Displays the contribution of each feature to the prediction."
    )

    st.markdown("---")

# ==========================================================
# PREDICTION PAGE
# ==========================================================

elif page == "Movie Success Prediction":

    st.header("Movie Success Prediction")

    st.write(
        """
        Enter the movie details below to predict whether
        the movie is likely to be successful.
        """
    )

    st.markdown("---")

    budget = st.number_input(
        "Budget",
        min_value=0.0,
        value=1000000.0,
        step=100000.0
    )

    popularity = st.number_input(
        "Popularity",
        min_value=0.0,
        value=20.0,
        step=1.0
    )

    runtime = st.number_input(
        "Runtime (minutes)",
        min_value=30.0,
        value=120.0,
        step=1.0
    )

    vote_average = st.slider(
        "Vote Average",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1
    )

    st.markdown("---")

    if st.button("Predict Movie Success"):

        prediction = model.predict([[
            budget,
            popularity,
            runtime,
            vote_average
        ]])

        prediction_probability = model.predict_proba([[
            budget,
            popularity,
            runtime,
            vote_average
        ]])

        confidence = prediction_probability.max() * 100

        st.markdown("---")

        if prediction[0] == 1:

            st.success("🎉 Prediction: This movie is likely to be SUCCESSFUL!")

        else:

            st.error("❌ Prediction: This movie is likely to be UNSUCCESSFUL.")

        st.info(f"Model Confidence: {confidence:.2f}%")

        st.markdown("---")

        st.subheader("Input Summary")

        summary = pd.DataFrame({
            "Feature": [
                "Budget",
                "Popularity",
                "Runtime",
                "Vote Average"
            ],
            "Value": [
                budget,
                popularity,
                runtime,
                vote_average
            ]
        })

        st.table(summary)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "MovieIQ | Predictive Analytics on Movie Success | "
    "Developed using Python, Streamlit, Scikit-learn, Pandas, and Matplotlib."
)



# In[ ]:




