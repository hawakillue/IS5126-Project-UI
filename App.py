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
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Car Market Singapore",
        page_icon="👋",
    )

    st.write("# Welcome to SG Car Market! 👋")

    st.sidebar.success("Select a function above.")

    st.image("car.jpg")

    st.markdown(
        """
        ### Carc Market Overview
        ### Car Price Evaluation
        Willing to sell your car but wonder about the price evaluation? 
        Provide the basic car information and get the estimated price in seconds with our app!
    """
    )


if __name__ == "__main__":
    run()
