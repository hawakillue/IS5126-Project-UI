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
import pandas as pd
import altair as alt
from utils import show_code

from urllib.error import URLError


st.set_page_config(page_title="Thanks", page_icon="📊")
st.markdown("# Special Thanks to Teammates: ")
st.sidebar.header("Special Thanks")
st.write(
    """
    \n
    **Zhang Shilin** \n
    **Lin Zhao** \n
    **Li Hao** \n
    **Xing Da** \n
    **Liu Yanli** \n
    """
)
