import base64
import streamlit as st

from app1 import app1_main
from app2 import app2_main
from app3 import app3_main

# Function to convert a file to Base64
def get_base64_of_file(file_path):
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode()

# Specify the correct path to your GIF file
file_path = "animated.gif"  # Adjust this path as needed
gif_base64 = get_base64_of_file(file_path)

# Streamlit app code
st.set_page_config(page_title="MultiModel Vision")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "App 1", "App 2", "App 3"])

# Add animated image to the center
if page == "Home":
    st.title("Welcome to the MultiModel Vision")
    st.write("⬅️Select an app from the dropdown menu.")
    
    # Embed the GIF using Base64
    st.markdown(
        f"""
        <style>
        .centered-image {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 300px; /* Adjust as needed */
        }}
        </style>
        <div class="centered-image">
            <img src="data:image/gif;base64,{gif_base64}" width="500" height="300" />
        </div>
        """,
        unsafe_allow_html=True
    )

elif page == "App 1":
    app1_main()
elif page == "App 2":
    app2_main()
elif page == "App 3":
    app3_main()
