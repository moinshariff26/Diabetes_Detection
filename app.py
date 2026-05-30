import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
cwd = os.getcwd()
file_path = os.path.join(cwd, 'diabetes.csv')
df = pd.read_csv(file_path)

# Prepare data
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train model
rf = RandomForestClassifier(random_state=0)
rf.fit(x_train, y_train)

# Save and load the model
model_path = os.path.join(cwd, 'rf_model.pkl')
joblib.dump(rf, model_path)
model = joblib.load(model_path)

# Custom Streamlit styles
st.set_page_config(page_title="Diabetes Prediction", layout="wide")
st.markdown("""
    <style>
    .reportview-container {
        background-color: #F4F4F4;
        padding: 20px;
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3, h4 {
        color: #4535C1;
    }
    .stButton > button {
        background-color: #36C2CE;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Introduction
st.title("Diabetes Prediction and Analysis Tool")
st.markdown("""
This tool leverages machine learning to predict diabetes risk and provides a detailed comparison of your health metrics 
with historical patient data. Explore insightful visualizations to better understand key health indicators.
""")

# Sidebar for Input
st.sidebar.header("Enter Patient Data")
pregnancies = st.sidebar.slider("Pregnancies", min_value=0, max_value=20, step=1, value=0)
glucose = st.sidebar.slider("Glucose Level", min_value=0.0, max_value=300.0, step=1.0, value=120.0)
blood_pressure = st.sidebar.slider("Blood Pressure", min_value=0.0, max_value=200.0, step=1.0, value=80.0)
skin_thickness = st.sidebar.slider("Skin Thickness", min_value=0.0, max_value=100.0, step=1.0, value=20.0)
insulin = st.sidebar.slider("Insulin", min_value=0.0, max_value=900.0, step=1.0, value=80.0)
bmi = st.sidebar.slider("BMI", min_value=0.0, max_value=70.0, step=0.1, value=25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01, value=0.5)
age = st.sidebar.slider("Age", min_value=0, max_value=120, step=1, value=30)

# Prediction
user_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]).reshape(1, -1)
prediction = model.predict(user_data)
output = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
accuracy = accuracy_score(y_test, model.predict(x_test)) * 100

st.sidebar.markdown("### Prediction Result")
st.sidebar.write(f"**{output}**")
st.sidebar.write(f"Model Accuracy: **{accuracy:.2f}%**")

# Main Section - Data Insights
st.header("Training Data Insights")
st.markdown("Here’s an overview of the dataset used to train the model.")
st.dataframe(df.describe())

# Visualization Function
def plot_comparison(user_value, column_name, title):
    plt.figure(figsize=(8, 4))
    sns.histplot(df[column_name], bins=30, kde=True, color="#478CCF", label="Dataset")
    plt.axvline(user_value, color="#4535C1", linestyle="--", linewidth=2, label="Your Value")
    plt.title(title, fontsize=14)
    plt.xlabel(column_name, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    st.pyplot(plt)

# Personalized Patient Report
st.header("Your Health Metrics vs Dataset")
st.write("Compare your values with historical patient data to understand your health better.")

cols = st.columns(2)

with cols[0]:
    plot_comparison(pregnancies, "Pregnancies", "Pregnancy Count (You vs Others)")
    plot_comparison(glucose, "Glucose", "Glucose Levels (You vs Others)")
    plot_comparison(blood_pressure, "BloodPressure", "Blood Pressure (You vs Others)")

with cols[1]:
    plot_comparison(skin_thickness, "SkinThickness", "Skin Thickness (You vs Others)")
    plot_comparison(insulin, "Insulin", "Insulin Levels (You vs Others)")
    plot_comparison(bmi, "BMI", "BMI (You vs Others)")

st.write("### Additional Comparisons")
plot_comparison(dpf, "DiabetesPedigreeFunction", "Diabetes Pedigree Function (You vs Others)")
plot_comparison(age, "Age", "Age (You vs Others)")

# Heatmap of Correlations
st.header("Feature Correlation Heatmap")
st.write("Explore the relationships between various health metrics.")
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
st.pyplot(plt)

# Outcome Distribution
st.header("Outcome Distribution")
st.write("Distribution of Diabetic vs Non-Diabetic patients in the dataset.")
plt.figure(figsize=(8, 4))
sns.countplot(x="Outcome", data=df, palette="Set2")
plt.title("Diabetes Outcome Distribution", fontsize=14)
plt.xlabel("Outcome", fontsize=12)
plt.ylabel("Count", fontsize=12)
st.pyplot(plt)
