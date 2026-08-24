import streamlit as st
from models import cardio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# Function implementation
features, scaler, model, Y_pred, cr, cm = cardio()

st.header('Cardiovascular Disease Prediction')
st.subheader('Using Logistic Regression')

API_URL = 'https://cardio-6.onrender.com/predict-logistic-cardio'

# Input Features 
# age = st.text_input('Age', placeholder='Enter your age')
# age = st.slider('Age', min_value=28, max_value = 65, value=35)

st.sidebar.header(
    'Cardio Features'
)

age = st.sidebar.slider(
    'Age', min_value=28, max_value = 65, value=35, step=1
)

gender_dict = {1: 'Female', 2: 'Male'}
gender = st.sidebar.radio(
    'Gender',
    options = list(gender_dict.keys()),
    format_func = lambda x : gender_dict.get(x)
)

height = st.sidebar.slider(
    'Height (Cm)', min_value=125, max_value=200, value=140, step=1
)

weight = st.sidebar.slider(
    'Weight (Kg)', min_value=40, max_value=120, value=60, step=1
)

ap_hi = st.sidebar.slider(
    'Systolic Pressure', min_value=100, max_value=200, value=120, step=1
)

ap_lo = st.sidebar.slider(
    'Di-Systolic Pressure', min_value=50, max_value=90, value=120, step=1
)

cholesterol_dict = {1: 'Low Cholesterol', 2: 'Mild Cholesterol', 3: 'High Cholesterol'}
cholesterol = st.sidebar.radio(
    'Cholesterol',
    options = list(cholesterol_dict.keys()),
    format_func = lambda x : cholesterol_dict.get(x)
)

gluc_dict = {1: 'Low Glucose', 2: 'Mild Glucose', 3: 'High Glucose'}
gluc = st.sidebar.radio(
    'Glucose',
    options = list(gluc_dict.keys()),
    format_func = lambda x : gluc_dict.get(x)
)

smoke_dict = {0: 'Doesnot Smoke', 1: 'Does Smoke'}
smoke = st.sidebar.radio(
    'Smoke', options = list(smoke_dict.keys()),
    format_func = lambda x : smoke_dict.get(x)
)

alco_dict = {0: "Doesn't drink alcohol", 1: 'Does drink alcohol'}
alco = st.sidebar.radio(
    'Alcohol', options = list(alco_dict.keys()),
    format_func = lambda x : alco_dict.get(x)
)

active_dict = {0: "Doesn't do PA", 1: 'Does do PA'}
active = st.sidebar.radio(
    'Physical Activities (PA)',
    options = list(active_dict.keys()),
    format_func = lambda x : active_dict.get(x)
)

# ## Create prediction button
# if st.button('Predict Cardio'):
#     data = pd.DataFrame([[
#         age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active
#     ]], columns=features)
#     data_scale = scaler.transform(data)
#     prediction = model.predict(data_scale)[0]
#     if prediction == 0:
#         st.success('No cardiovascular disease found.')
#     else:
#         st.warning('Cardiovascular disease found.')

if st.button('Predict Cardio'):
    payload = {
        'age': age,
        'gender': gender,
        'height': height,
        'weight': weight,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'cholesterol': cholesterol,
        'gluc': gluc,
        'smoke': smoke,
        'alco': alco,
        'active': active
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            if result['Prediction Status'] == 0:
                st.write('Likely to be healthy')
                st.success('No cardiovasular disease found.')
            else:
                st.write('Likely to be unhealthy')
                st.success('Cardiovasular disease found.')
        else:
            st.error(f'API Status Code Error: {response.status_code}')
    except requests.exceptions.RequestException as e:
        st.error(f'API Error: {e}')


        

# Visualization
st.subheader('Visualization')
fig, axes = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='1.0f', 
            xticklabels = ['Predicted Healthy[0]', 'Predicted Unhealthy[1]'],
           yticklabels = ['Actual Healthy[0]', 'Actual Unhealthy[1]'])
plt.title('Actual Cardio vs. Predicted Cardio')
st.pyplot(fig)


### Classification Report
st.subheader('Classification Report')
data = pd.DataFrame(cr).transpose()
st.dataframe(data.style.format(precision=2))