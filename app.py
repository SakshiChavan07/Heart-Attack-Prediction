import streamlit as st
import numpy as np
import pickle

# -----------------------
# Load Trained Model
# -----------------------

# Load your saved model (replace 'your_model.pkl' with your actual filename)
with open('your_model.pkl', 'rb') as file:
    model = pickle.load(file)

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
    st.success(f"Hello {name}, Let's check your heart health. ❤️!")

# Input fields
age = st.number_input('Enter your Age:', min_value=1, max_value=120)
heart_rate = st.number_input('Enter your Heart Rate:', min_value=30, max_value=220)
blood_sugar = st.number_input('Enter your Blood Sugar Level:', min_value=50, max_value=300)
smoking = st.selectbox('Do you smoke?', ['No', 'Yes'])

# Preprocessing smoking input (if needed)
smoking_value = 1 if smoking == 'Yes' else 0

# Combine all inputs (if your model expects smoking too, adjust accordingly)
input_data = np.array([[age, heart_rate, blood_sugar, smoking_value]])

# Prediction
if st.button('Predict Heart Attack Risk'):
    prediction = model.predict(input_data)

    if prediction[0] == 2:
        st.error('⚠️ High Risk of Heart Attack! Please consult a doctor.')
        st.write("### Health Tips for High Risk:")
        st.write("- Maintain a healthy diet low in fats and sugars.")
        st.write("- Exercise regularly to improve cardiovascular health.")
        st.write("- Monitor blood pressure and cholesterol levels.")
        st.write("- Avoid smoking and manage stress.")
    elif prediction[0] == 1:
        st.warning('⚠️ Moderate Risk of Heart Attack. Monitor closely and consult a doctor.')
        st.write("### Health Tips for Moderate Risk:")
        st.write("- Regular checkups are important.")
        st.write("- Keep track of your blood pressure and cholesterol levels.")
        st.write("- Exercise and maintain a healthy diet.")
    else:
        st.success('✅ Low Risk of Heart Attack. Stay healthy!')
        st.write("### Health Tips for Low Risk:")
        st.write("- Keep up with regular physical activity.")
        st.write("- Follow a balanced and nutritious diet.")
        st.write("- Stay hydrated and maintain a healthy weight.")
        st.write("- Regular checkups are still important.")

st.write('---')
st.caption('Developed by Sakshi Chavan')
