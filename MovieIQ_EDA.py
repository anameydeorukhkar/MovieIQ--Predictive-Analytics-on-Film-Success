#!/usr/bin/env python
# coding: utf-8

# In[22]:


# ==========================
# Import Required Libraries
# ==========================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

import joblib
import ast

# Set graph style

sns.set_style("whitegrid")

plt.rcParams["figure.figsize"] = (10,6)


# In[4]:


# ==========================
# Load Dataset
# ==========================

movies = pd.read_csv(r"C:\Users\Anamey\Downloads\movies.csv")
movies.head()


# In[5]:


movies.shape


# In[6]:


movies.info()


# In[7]:


movies.describe()


# In[8]:


movies.isnull().sum()


# In[9]:


movies.duplicated().sum()


# In[10]:


print("Movies with zero budget :", (movies["budget"] == 0).sum())

print("Movies with zero revenue :", (movies["revenue"] == 0).sum())


# In[11]:


movies = movies[
    (movies["budget"] > 0) &
    (movies["revenue"] > 0)
]


# In[12]:


movies.shape


# In[13]:


movies["success"] = (
    movies["revenue"] > movies["budget"]
).astype(int)


# In[14]:


movies[["budget","revenue","success"]].head()


# In[15]:


movies["success"].value_counts()


# In[17]:


success_percentage = (
    movies["success"]
    .value_counts(normalize=True)
    * 100
)

print(success_percentage)


# In[18]:


def extract_genres(genre_string):
    genre_list = ast.literal_eval(genre_string)

    names = [
        genre["name"]
        for genre in genre_list
    ]

    return ", ".join(names)


# In[19]:


movies["genres"] = movies["genres"].apply(extract_genres)
movies[["title","genres"]].head()


# In[20]:


movies.info()


# In[21]:


movies.to_csv(
    "clean_movies.csv",
    index=False
)


# In[44]:


plt.figure(figsize=(10,6))

sns.scatterplot(
    data=movies,
    x="budget",
    y="revenue"
)

plt.title("Budget vs Revenue")
plt.xlabel("Budget")
plt.ylabel("Revenue")

plt.show()


# In[46]:


plt.figure(figsize=(10,6))

genre_counts.plot(kind="bar", color="steelblue")

plt.title("Most Common Movie Genres")
plt.xlabel("Genre")
plt.ylabel("Number of Movies")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("assets/genre_frequency.png",
            dpi=300,
            bbox_inches="tight")

plt.show()


# In[26]:


genre_success = movies.copy()

genre_success["genres"] = (
    genre_success["genres"]
    .str.split(", ")
)

genre_success = genre_success.explode("genres")

genre_success_rate = (
    genre_success
    .groupby("genres")["success"]
    .mean()
    .sort_values(ascending=False)
)


# In[47]:


plt.figure(figsize=(10,6))

genre_success_rate.plot(kind="bar", color="seagreen")

plt.title("Success Rate by Genre")
plt.xlabel("Genre")
plt.ylabel("Average Success")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("assets/genre_success_rate.png",
            dpi=300,
            bbox_inches="tight")

plt.show()


# In[48]:


plt.figure(figsize=(8,6))

sns.boxplot(
    data=movies,
    x="success",
    y="popularity"
)

plt.title("Popularity vs Movie Success")
plt.xlabel("Success")
plt.ylabel("Popularity")

plt.tight_layout()

plt.savefig("assets/popularity_vs_success.png",
            dpi=300,
            bbox_inches="tight")

plt.show()


# In[49]:


plt.figure(figsize=(8,6))

sns.boxplot(
    data=movies,
    x="success",
    y="runtime"
)

plt.title("Runtime vs Movie Success")
plt.xlabel("Success")
plt.ylabel("Runtime")

plt.tight_layout()

plt.savefig("assets/runtime_vs_success.png",
            dpi=300,
            bbox_inches="tight")

plt.show()


# In[50]:


plt.figure(figsize=(8,6))

sns.boxplot(
    data=movies,
    x="success",
    y="vote_average"
)

plt.title("Vote Average vs Movie Success")
plt.xlabel("Success")
plt.ylabel("Vote Average")

plt.tight_layout()

plt.savefig("assets/vote_average_vs_success.png",
            dpi=300,
            bbox_inches="tight")

plt.show()


# In[51]:


plt.figure(figsize=(10,8))

numeric_columns = movies.select_dtypes(include=["number"])

sns.heatmap(
    numeric_columns.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("assets/correlation_heatmap.png",
            dpi=300,
            bbox_inches="tight")

plt.show()


# In[52]:


plt.figure(figsize=(7,5))

sns.countplot(
    data=movies,
    x="success"
)

plt.title("Distribution of Successful Movies")
plt.xlabel("Success")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig("assets/success_distribution.png",
            dpi=300,
            bbox_inches="tight")

plt.show()


# In[36]:


import os

os.makedirs("assets", exist_ok=True)
os.makedirs("models", exist_ok=True)


# In[53]:


import os

print(os.listdir("assets"))


# In[57]:


X = movies[[
    "budget",
    "popularity",
    "runtime",
    "vote_average"
]]

y = movies["success"]


# In[58]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# In[59]:


rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# In[60]:


rf_model.fit(X_train, y_train)


# In[62]:


y_pred = rf_model.predict(X_test)


# In[63]:


prediction_df = X_test.copy()

prediction_df["Actual"] = y_test.values
prediction_df["Predicted"] = y_pred

prediction_df.head()


# In[65]:


accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2%}")


# In[66]:


cm = confusion_matrix(y_test, y_pred)

print(cm)


# In[67]:


plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Failure", "Success"],
    yticklabels=["Failure", "Success"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "assets/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[68]:


print(classification_report(y_test, y_pred))


# In[69]:


feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance


# In[70]:


plt.figure(figsize=(8,5))

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig(
    "assets/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[71]:


joblib.dump(
    rf_model,
    "models/rf_model.pkl"
)


# In[72]:


print(os.listdir("models"))


# In[ ]:




