import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure the generative AI model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to get Gemini response
def get_gemini_response(input_text, image_data, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input_text, image_data[0], prompt])
    return response.text

# Function to process the uploaded image
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Main function for app2
def app2_main():
    st.header("Image Insight🔍")

    # Input text prompt
    input_text = st.text_input("Input Prompt: ", key="input_app2")

    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Display uploaded image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Button to submit and get response
    submit = st.button("Tell me about the image", key="submit_app2")

    input_prompt = """
    As an expert in invoice analysis with a keen eye for detail and a comprehensive understanding of financial documents, your task is to review and interpret input images of invoices. Based on these images, you are required to accurately answer specific questions and provide insights. Your expertise should cover the following:
    Invoice Details: Extract and verify key information such as invoice number, date, supplier details, and payment terms.
    Itemization and Charges: Analyze the itemized list of products or services, including quantities, unit prices, taxes, and total amounts.
    Discrepancies and Anomalies: Identify any discrepancies, errors, or anomalies in the invoice data that may require attention.
    Overall Assessment: Provide a thorough evaluation of the invoice's accuracy and completeness, including any additional observations or recommendations.
    Your in-depth knowledge and analytical skills are crucial in ensuring accurate and efficient processing of these invoices.
    """

    # Handle button click
    if submit and uploaded_file is not None:
        image_data = input_image_setup(uploaded_file)
        response = get_gemini_response(input_text, image_data, input_prompt)
        st.subheader("The Response is")
        st.write(response)
