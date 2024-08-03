import streamlit as st
from app1 import app1_main
from app2 import app2_main
from app3 import app3_main
from app4 import app4_main  # Import app4

# Set page config only once in the main script
st.set_page_config(page_title="MultiModel Vision")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "App 1", "App 2", "App 3", "App 4"])

# Navigation logic
if page == "Home":
    st.title("Welcome to the MultiModel Vision")
    st.write("⬅️Select an app from the dropdown menu.")
elif page == "App 1":
    app1_main()
elif page == "App 2":
    app2_main()
elif page == "App 3":
    app3_main()
elif page == "App 4":
    app4_main()  # Call the main function for app4
