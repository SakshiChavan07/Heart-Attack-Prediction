"""
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
# User input: Name and Gender
name = st.text_input('Enter your Name:')
gender = st.selectbox('Select your Gender:', ['Male', 'Female', 'Other'])

# Greeting message when both fields are filled
if name and gender:
    st.success(f"Hello {name}, let's get started!")
    
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
st.caption('Developed by Sakshi Chavan')
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
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

# User input: Name and Gender
name = st.text_input('Enter your Name:')
gender = st.selectbox('Select your Gender:', ['Male', 'Female', 'Other'])

# Greeting message when both fields are filled
if name and gender:
    st.success(f"Hello {name}, let's get started! You are a {gender}.")

# Input fields for health details
age = st.number_input('Enter your Age:', min_value=1, max_value=120)
heart_rate = st.number_input('Enter your Heart Rate:', min_value=30, max_value=220)
blood_sugar = st.number_input('Enter your Blood Sugar Level:', min_value=50, max_value=300)

# BMI Calculator
st.write("### BMI Calculator")
weight = st.number_input('Enter your Weight (kg):', min_value=20, max_value=200)
height = st.number_input('Enter your Height (cm):', min_value=50, max_value=250)

if weight and height:
    bmi = weight / (height / 100) ** 2
    st.write(f'Your BMI is: {bmi:.2f}')
    if bmi < 18.5:
        st.warning('Underweight - Consider gaining weight for better health.')
    elif 18.5 <= bmi < 24.9:
        st.success('Normal weight - Keep maintaining your healthy lifestyle.')
    elif 25 <= bmi < 29.9:
        st.warning('Overweight - It is advised to maintain a balanced diet and exercise regularly.')
    else:
        st.error('Obesity - Seek advice from a healthcare provider for weight management.')

# Prediction
if st.button('Predict Heart Attack Risk'):
    input_data = np.array([[age, heart_rate, blood_sugar]])
    prediction = model.predict(input_data)
    
    # Visualization of Health Parameters
    fig, ax = plt.subplots()
    ax.bar(['Age', 'Heart Rate', 'Blood Sugar'], [age, heart_rate, blood_sugar], color=['blue', 'orange', 'green'])
    st.pyplot(fig)
    
    if prediction[0] == 1:
        st.error('⚠️ High Risk of Heart Attack! Please consult a doctor.')
        st.write("### Health Tips for High Risk:")
        st.write("- Maintain a healthy diet low in fats and sugars.")
        st.write("- Exercise regularly to improve cardiovascular health.")
        st.write("- Monitor blood pressure and cholesterol levels.")
        st.write("- Avoid smoking and manage stress.")
    else:
        st.success('✅ Low Risk of Heart Attack. Stay healthy!')
        st.write("### Health Tips for Low Risk:")
        st.write("- Keep up with regular physical activity.")
        st.write("- Follow a balanced and nutritious diet.")
        st.write("- Stay hydrated and maintain a healthy weight.")
        st.write("- Regular checkups are still important.")

st.write('---')
st.caption('Developed by [Your Name]')
