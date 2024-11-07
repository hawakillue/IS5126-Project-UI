import streamlit as st
import inspect
import textwrap
import pandas as pd

def show_code(demo):
    """Showing the code of the demo."""
    show_code = st.sidebar.checkbox("Show code", True)
    if show_code:
        # Showing the code of the demo.
        st.markdown("## Code")
        sourcelines, _ = inspect.getsourcelines(demo)
        st.code(textwrap.dedent("".join(sourcelines[1:])))


columns_needed = [
    'Car Brand', 'Car Model', 'COE', 'Price', 'Dereg Value', 'Manufactured',
    'No. of Owners', 'Mileage per year', 'Engine Capacity', 'Engine Type',
    'Category_Multilabel'
]

file_path = 'filled_car_data.csv'

filled_car_data = pd.read_csv(file_path, dtype={'Category_Multilabel': str})
model_car_data = filled_car_data[columns_needed]
model_car_data.dropna(inplace=True)

car_brands = model_car_data['Car Brand'].unique().tolist()
car_models = model_car_data['Car Model'].unique().tolist()
engine_types = model_car_data['Engine Type'].unique().tolist()
categories = ['Hybrid Cars',
'Direct Owner Sale',
'COE Car',
'Sgcarmart Warranty Cars',
'Direct Owner Sale',
'Almost New Car',
'Rare & Exotic',
'Electric Cars',
'Low Mileage Car',
'Consignment Car',
'OPC Car',
'PARF Car',
'Premium Ad Car',
'Rare & Exotic',
'-'
]

def getCategory(selected_category, all_category:list[str]):
    result = '000000000000000'
    for category in selected_category:
        index = all_category.index(category)
        result = result[:index] + '1' + result[index+1:]
    return result