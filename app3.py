from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure the generative AI model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Gemini Pro model
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

# Function to get response from Gemini Pro
def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

# Main function for app3
def app3_main():
    st.header("Ask Me Anything🫂")

    # Initialize session state for chat history if not already present
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # User input for question
    input_text = st.text_input("Input:", key="input_app3")
    submit = st.button("Ask the question", key="submit_app3")

    # Handle the submit action
    if submit and input_text:
        response = get_gemini_response(input_text)
        # Add user query to chat history
        st.session_state['chat_history'].append(("You", input_text))
        st.subheader("The response is")
        # Display the response and add to chat history
        for chunk in response:
            st.write(chunk.text)
            st.session_state['chat_history'].append(("Bot", chunk.text))

    # Display chat history
    st.subheader("The Chat History is")
    for role, text in st.session_state['chat_history']:
        st.write(f"{role}: {text}")
