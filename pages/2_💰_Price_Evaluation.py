import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from utils import car_brands, car_models, engine_types, categories,getCategory
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from joblib import load
from sklearn.preprocessing import MultiLabelBinarizer


st.set_page_config(page_title="Mapping Demo", page_icon="💰")
st.markdown("# Mapping Demo")
st.sidebar.header("Mapping Demo")
st.title("Car Price Prediction with Voting Ensemble Model")
st.write("""
This application allows you to predict the car price based on selected features. 
Please provide the necessary input values below.
""")
st.header("Car Details")

voting_ensemble = joblib.load('voting_ensemble_model.pkl')
label_encoders = load('label_encoders.joblib')
scaler = load('scaler.joblib')

car_brand = st.selectbox("Car Brand", car_brands)
car_model = st.selectbox("Car Model", car_models)
coe_value = st.number_input("COE Value")
dereg_value = st.number_input("Dereg Value")
manufactured_year = st.number_input("Manufactured Year", min_value=2000, max_value=datetime.now().year)
no_of_owners = st.number_input("No. of Owners", min_value=0, max_value=5)
mileage_per_year = st.number_input("Mileage per year")
engine_capacity = st.number_input("Engine Capacity", step=0.1)
engine_type = st.selectbox("Engine Type", engine_types)

selected_tags = st.multiselect("Categories", categories)
# selected_tags_as_string_array = ','.join(selected_tags)


def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data])

    current_year = 2024
    input_df['Manufactured'] = pd.to_numeric(input_df['Manufactured'], errors='coerce')
    input_df['Dereg Value'] = input_df['Dereg Value'].replace(r'[^\d.]+', '', regex=True).astype(float)
    input_df['No. of Owners'] = pd.to_numeric(input_df['No. of Owners'], errors='coerce')
    input_df['Engine Capacity'] = input_df['Engine Capacity'].replace(r'[^\d.]', '', regex=True).astype(float)
    
    input_df['Car_Age'] = current_year - input_df['Manufactured']
    input_df['Log_COE'] = np.log1p(input_df['COE'])
    input_df['Log_Dereg_Value'] = np.log1p(input_df['Dereg Value'])
    input_df['Log_Mileage_per_year'] = np.log1p(input_df['Mileage per year'])
    input_df['Car_Age_COE'] = input_df['Car_Age'] * input_df['Log_COE']
    print(input_df)

    categorical_columns = ['Car Brand', 'Car Model', 'Engine Type', 'Category_Multilabel']
    le = LabelEncoder()
    for col in categorical_columns:
        input_df[col] = le.fit_transform(input_df[col].astype(str))

    input_df.drop(columns=['Manufactured', 'COE', 'Dereg Value', 'Mileage per year'], inplace=True)
    print(input_df)

    scaler = StandardScaler()
    input_df_scaled = voting_ensemble.transform(input_df)  
    # input_df_scaled = voting_ensemble.transform(input_df)
    print(input_df_scaled)

    return input_df_scaled

user_input = {
    'Car Brand': car_brand,   
    'Car Model': car_model,
    'COE': coe_value,
    'Dereg Value': dereg_value,
    'Manufactured': manufactured_year,
    'No. of Owners': no_of_owners,
    'Mileage per year': mileage_per_year,
    'Engine Capacity': engine_capacity,
    'Engine Type': engine_type,
    # 'Category_Multilabel': '000000001001100'
    'Category_Multilabel': selected_tags
}


def predict_with_ensemble(test_data, label_encoders, scaler, voting_ensemble, current_year=2024):
    test_data = pd.DataFrame([test_data])
    print('test_data')
    print(test_data)

    necessary_columns = ['Manufactured', 'Dereg Value', 'No. of Owners', 'Engine Capacity']
    for col in necessary_columns:
        if col not in test_data.columns:
            raise ValueError(f"Missing necessary column in input data: {col}")
  

    test_data['Manufactured'] = pd.to_numeric(test_data['Manufactured'], errors='coerce')
    test_data['Dereg Value'] = test_data['Dereg Value'].replace(r'[^\d.]+', '', regex=True).astype(float)
    test_data['No. of Owners'] = pd.to_numeric(test_data['No. of Owners'], errors='coerce')
    test_data['Engine Capacity'] = test_data['Engine Capacity'].replace(r'[^\d.]', '', regex=True).astype(float)
    test_data['Car_Age'] = current_year - test_data['Manufactured']
    test_data['Log_COE'] = np.log1p(test_data['COE'])
    test_data['Log_Dereg_Value'] = np.log1p(test_data['Dereg Value'])
    test_data['Log_Mileage_per_year'] = np.log1p(test_data['Mileage per year'])
    test_data['Car_Age_COE'] = test_data['Car_Age'] * test_data['Log_COE']

    print('lalala')
    print(test_data['Category_Multilabel'])
    test_data['Category_Multilabel'] = getCategory(selected_tags, categories)
    print('now category multilabel is: ')
    print(test_data['Category_Multilabel'])
    
    for col in label_encoders:
        test_data[col] = label_encoders[col].transform(test_data[col].astype(str))

    
    features = [col for col in test_data.columns if col in label_encoders or col in ['Car_Age', 'Log_COE', 'Log_Dereg_Value', 'Log_Mileage_per_year', 'Car_Age_COE', 'Engine Capacity', 'No. of Owners']]
    test_data = test_data[features]

    
    test_data_scaled = scaler.transform(test_data)

    # predict with ensemble model
    prediction = voting_ensemble.predict(test_data_scaled)
    return prediction



submitted = st.button('Submit')

if submitted:
    with st.spinner("Calculating..."):
        time.sleep(1)
        predicted_price = predict_with_ensemble(user_input, label_encoders, scaler, voting_ensemble)
        st.balloons()  
        st.markdown(f"<h2 style='color:blue;'>Estimated Price: ${predicted_price[0]}</h2>", unsafe_allow_html=True)