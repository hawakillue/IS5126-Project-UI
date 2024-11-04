import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from utils import car_brands, car_models, engine_types
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np


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

car_brand = st.selectbox("Car Brand", car_brands)
car_model = st.selectbox("Car Model", car_models)
coe_value = st.number_input("COE Value")
dereg_value = st.number_input("Dereg Value")
manufactured_year = st.number_input("Manufactured Year", min_value=2000, max_value=datetime.now().year)
no_of_owners = st.number_input("No. of Owners", min_value=0, max_value=5)
mileage_per_year = st.number_input("Mileage per year")
engine_capacity = st.number_input("Engine Capacity", step=0.1)
engine_type = st.selectbox("Engine Type", engine_types)


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

    categorical_columns = ['Car Brand', 'Car Model', 'Engine Type', 'Category_Multilabel']
    le = LabelEncoder()
    for col in categorical_columns:
        input_df[col] = le.fit_transform(input_df[col].astype(str))

    input_df.drop(columns=['Manufactured', 'COE', 'Dereg Value', 'Mileage per year'], inplace=True)
    print(input_df)

    scaler = StandardScaler()
    # input_df_scaled = scaler.fit_transform(input_df)  
    input_df_scaled = voting_ensemble.transform(input_df)
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
    'Category_Multilabel': '000000001001100'
}



submitted = st.button('Submit')

if submitted:
    processed_input = preprocess_input(user_input)
    predicted_price = voting_ensemble.predict(processed_input)
    # 显示预测结果
    st.write(f"预测价格: {predicted_price[0]}")
