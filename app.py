import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from PIL import Image

# --- Page Configuration ---
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

# --- Cached functions for performance ---
@st.cache_resource
def load_model():
    return pickle.load(open('titanicmodel.pkl', 'rb'))

@st.cache_data
def load_data():
    return pd.read_csv('titanic.csv')

model = load_model()
df = load_data()


# --- Title and Image ---
st.markdown("<h1 style='color: #0E76A8;'>🛳 Titanic Survival Predictor App</h1>", unsafe_allow_html=True)
image = Image.open('OIP.webp')
st.image(image, caption="Titanic - The Unsinkable Ship", use_container_width=True)

st.markdown("---")  # Divider

st.markdown("### ✍️ Brief Introduction")
st.markdown("""
The **RMS Titanic** was a British passenger liner that tragically sank on its maiden voyage in April 1912, resulting in the deaths of over 1,500 people.  
Despite being labeled as "unsinkable", the ship struck an iceberg, revealing the lack of sufficient lifeboats and safety measures on board.

This web application utilizes a **Logistic Regression** model, trained in **Google Colab**, to estimate the survival chances of passengers based on their details such as passenger class, age, gender, fare paid, and port of embarkation.

The aim of this project is not only to apply machine learning for predictive analysis but also to understand the human and data patterns behind a historic tragedy.  
It provides a simple yet interactive way to explore how various factors influenced survival on the Titanic.
""")



st.markdown("---")  # Divider

# --- Data Dictionary ---
st.markdown("<h3 style='color: #6A5ACD;'>📖 Data Dictionary</h3>", unsafe_allow_html=True)
st.markdown("""
| Variable   | Definition                          | Key |
|------------|-------------------------------------|-----|
| survived   | Survival status (Target variable)   | 0 = No, 1 = Yes |
| pclass     | Ticket class                        | 1 = 1st, 2 = 2nd, 3 = 3rd |
| sex        | Gender of passenger                 | male / female |
| age        | Age of passenger                    | - |
| sibsp      | No. of siblings/spouses aboard      | - |
| parch      | No. of parents/children aboard      | - |
| fare       | Ticket fare paid                    | - |
| embarked   | Port of Embarkation                 | C = Cherbourg, Q = Queenstown, S = Southampton |
""")

st.markdown("---")  # Divider

# --- Visualizations ---
st.markdown("<h3 style='color: #2E8B57;'>📊 Titanic Data Visualizations</h3>", unsafe_allow_html=True)

# 1. Survival by Class
st.markdown("### 🎫 Survival Rate by Passenger Class")
fig1, ax1 = plt.subplots()
sns.barplot(x='pclass', y='survived', data=df, ax=ax1, palette='Blues')
ax1.set_title("Survival Rate by Passenger Class")
st.pyplot(fig1)

st.markdown("---")  # Divider

# 2. Survival by Gender
st.markdown("### 👥 Survival Rate by Gender")
fig2, ax2 = plt.subplots()
sns.barplot(x='sex', y='survived', data=df, ax=ax2, palette='Purples')
ax2.set_title("Survival Rate by Gender")
st.pyplot(fig2)

st.markdown("---")  # Divider

# 3. Age Distribution
st.markdown("### 🎂 Age Distribution of Passengers")
fig3, ax3 = plt.subplots()
sns.histplot(df['age'], bins=10, kde=True, color="green", ax=ax3)
ax3.set_title("Passenger Age Distribution")
st.pyplot(fig3)

st.markdown("---")  # Divider

# --- Passenger Input Form ---
st.markdown("<h3 style='color: #FF6347;'>🧾 Enter Passenger Details to Predict Survival</h3>", unsafe_allow_html=True)

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["Male", "Female"])
age = st.slider("Age", 1, 80, value=25)
sibsp = st.slider("Siblings/Spouses Aboard (SibSp)", 0, 5, value=0)
parch = st.slider("Parents/Children Aboard (Parch)", 0, 5, value=0)
fare = st.number_input("Fare Paid", min_value=0.0, value=50.0)
embarked = st.selectbox("Port of Embarkation", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"])

# Encode Inputs
sex_encoded = 1 if sex == "Male" else 0
embarked_c = 1 if embarked == "Cherbourg (C)" else 0
embarked_q = 1 if embarked == "Queenstown (Q)" else 0
# Southampton is implied when both are 0

input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_c, embarked_q]])

# --- Prediction Button ---
if st.button("🚀 Predict Survival"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.markdown("---")  # Divider

    if prediction[0] == 1:
        st.markdown("""
        <div style="background-color:#d4edda; padding: 15px; border-radius: 8px;">
            <h4 style='color:green;'>🎉 Prediction Results</h4>
            <p style='color:black; font-size:16px;'>You would most likely <b>SURVIVE</b> the Titanic disaster like <b>Rose 🌹</b>!</p>
            <p style='color:#155724;'><b>Survival Probability:</b> {:.2f}%</p>
        </div>
        """.format(probability[0][1]*100), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color:#f8d7da; padding: 15px; border-radius: 8px;">
            <h4 style='color:crimson;'>☠️ Prediction Results</h4>
            <p style='color:black; font-size:16px;'>You are most likely <b>NOT to survive</b> and end up like <b>Jack 🧊</b>.</p>
            <p style='color:#721c24;'><b>Survival Probability:</b> {:.2f}%</p>
        </div>
        """.format(probability[0][0]*100), unsafe_allow_html=True)
