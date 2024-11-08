# Copyright 2018-2022 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import inspect
import textwrap
import time
import numpy as np
from utils import show_code
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def plotting_demo():
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    last_rows = np.random.randn(1, 1)
    chart = st.line_chart(last_rows)

    for i in range(1, 101):
        new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
        status_text.text("%i%% Complete" % i)
        chart.add_rows(new_rows)
        progress_bar.progress(i)
        last_rows = new_rows
        time.sleep(0.05)

    progress_bar.empty()

    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    st.button("Re-run")


# Set up the page
st.set_page_config(page_title="SG Car Market Overview", page_icon="📈")
st.sidebar.header("SG Car Market Overview")

# Title
st.title("SG Car Market Overview")
st.write("An overview of the car market in Singapore based on the latest available data.")

# Key Market Statistics
st.subheader("Key Market Statistics")
st.markdown("**Total Car Population (2023):** 636,483 cars")
st.markdown("**Average Age of Cars:** 5.96 years")
st.markdown("**New Registrations (First Half of 2024):** 20,000 cars")
st.markdown("**Average Car Price:** SGD 120,000")

# Price Distribution
st.subheader("Price Distribution")
# Note: Replace the following line with actual price data when available
prices = [80000, 90000, 100000, 110000, 120000, 130000, 140000, 150000, 160000, 170000]
fig, ax = plt.subplots()
sns.histplot(prices, bins=10, kde=True, ax=ax)
ax.set_title("Distribution of Car Prices")
ax.set_xlabel("Price (SGD)")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Mileage Distribution
st.subheader("Mileage Distribution")
# Note: Replace the following line with actual mileage data when available
mileages = [10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000]
fig, ax = plt.subplots()
sns.histplot(mileages, bins=10, kde=True, ax=ax)
ax.set_title("Distribution of Car Mileages")
ax.set_xlabel("Mileage (km)")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Average Price by Car Brand
st.subheader("Average Price by Car Brand")
# Note: Replace the following dictionary with actual average prices when available
brand_prices = {
    "Toyota": 100000,
    "Honda": 95000,
    "Mercedes-Benz": 150000,
    "BMW": 140000,
    "Mazda": 90000
}
brands = list(brand_prices.keys())
prices = list(brand_prices.values())
fig, ax = plt.subplots()
sns.barplot(x=prices, y=brands, ax=ax)
ax.set_title("Average Price by Car Brand")
ax.set_xlabel("Average Price (SGD)")
ax.set_ylabel("Brand")
st.pyplot(fig)

plotting_demo()

show_code(plotting_demo)
