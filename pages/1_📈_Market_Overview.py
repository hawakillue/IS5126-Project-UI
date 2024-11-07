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


def plotting_demo():
    # Load CSV data
    coe_df = pd.read_csv('./COE.csv')

    # Convert 'month' to datetime for proper time series handling
    coe_df['month'] = pd.to_datetime(coe_df['month'])

    # Group by month and vehicle_class, taking the mean of 'premium' to handle duplicates
    coe_df = coe_df.groupby(['month', 'vehicle_class']).agg({'premium': 'mean'}).reset_index()

    # Pivot the DataFrame to have each vehicle_class as a separate column
    pivot_df = coe_df.pivot(index='month', columns='vehicle_class', values='premium')

    # Initialize progress bar and status text
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    # Initialize the chart with an empty DataFrame
    st.subheader("Trend of COE for Each Vehicle Class")
    chart = st.line_chart(pivot_df.iloc[:1])  # Start with the first row for correct columns setup

    # Iteratively add each row to the chart
    for i in range(1, len(pivot_df)):
        # Add the next row to the chart
        chart.add_rows(pivot_df.iloc[[i]])

        # Update the progress bar and status
        progress = int((i / (len(pivot_df) - 1)) * 100)
        status_text.text(f"{progress}% Complete")
        progress_bar.progress(progress)


    # Clear the progress bar after completion
    progress_bar.empty()
    st.button("Re-run")


st.set_page_config(page_title="SG Car Market Overview", page_icon="📈")
st.markdown("# SG Car Market Overview")
st.sidebar.header("SG Car Market Overview")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

plotting_demo()

show_code(plotting_demo)
