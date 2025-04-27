# app.py
import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# -----------------------
# Model Training
# -----------------------

# Dummy dataset for training
X = np.random.rand(200, 3)  # Features: Age, Heart Rate, Blood Sugar
y = np.random.randint(0, 2, 200)  # Labels: 0 = Low Risk, 1 = High Risk

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a simple Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------
# Streamlit App UI
# -----------------------

st.set_page_config(page_title="Heart Attack Risk Predictor", page_icon="❤️")
st.title('❤️ Heart Attack Risk Prediction App')

st.write("""
### Enter your health details below:
""")

# Input fields
age = st.number_input('Enter your Age:', min_value=1, max_value=120)
heart_rate = st.number_input('Enter your Heart Rate:', min_value=30, max_value=220)
blood_sugar = st.number_input('Enter your Blood Sugar Level:', min_value=50, max_value=300)

# Prediction
if st.button('Predict Heart Attack Risk'):
    input_data = np.array([[age, heart_rate, blood_sugar]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error('⚠️ High Risk of Heart Attack! Please consult a doctor.')
    else:
        st.success('✅ Low Risk of Heart Attack. Stay healthy!')

st.write('---')
st.caption('Developed by [Your Name]')
