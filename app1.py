from dotenv import load_dotenv
import streamlit as st
import os
from pdfminer.high_level import extract_text
import google.generativeai as genai

load_dotenv()

# Configure the generative AI model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(pdf_content, job_description, prompt):
    # Combine the job description with the extracted resume content and prompt
    input_text = f"Job Description:\n{job_description}\n\nResume Content:\n{pdf_content}\n\n{prompt}"
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(input_text)
    return response.text

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        # Extract text from the PDF
        text = extract_text(uploaded_file)
        return text
    else:
        raise FileNotFoundError("No file uploaded")

def app1_main():
    st.header("ATS Tracking System")

    input_text = st.text_area("Job Description: ", key="input")
    uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=["pdf"])

    if uploaded_file is not None:
        st.write("PDF Uploaded Successfully")

    submit1 = st.button("Tell Me About the Resume")
    submit3 = st.button("Percentage Match")

    input_prompt1 = """
    As a seasoned Technical Human Resource Manager with expertise in assessing technical and non-technical qualifications, your task is to meticulously review the provided resume in light of the accompanying job description. Your professional evaluation should focus on the following:
    Alignment with Job Requirements: Assess how well the candidate's skills, experience, and qualifications match the specific criteria outlined in the job description.
    Key Strengths: Identify the candidate's notable strengths, such as technical expertise, relevant experience, soft skills, or any unique attributes that make them a strong contender for the role.
    Areas for Improvement: Highlight any potential gaps or weaknesses in the candidate's profile, including missing qualifications, lack of experience in critical areas, or other factors that might impact their suitability for the position.
    Overall Fit: Provide an overall assessment of the candidate's fit for the role, considering both their potential contributions and any limitations.
    Your insights will play a crucial role in determining the candidate's suitability for the position, and we appreciate your thorough and expert analysis.


    """

    input_prompt3 = """
    As a proficient ATS (Applicant Tracking System) scanner with an in-depth understanding of data science and ATS functionalities, your task is to evaluate the provided resume against the accompanying job description. Please follow these steps:
    Percentage Match: Begin by calculating and providing a percentage score indicating how well the resume aligns with the job description.
    Missing Keywords: Identify and list any important keywords or key phrases from the job description that are missing from the resume. These keywords are crucial for assessing the candidate's fit for the role.
    Analysis and Thoughts: Offer a comprehensive analysis based on the resume's content, including both its strengths and any potential gaps. Provide your thoughts on the overall alignment and suggest areas for improvement, if applicable.
    Your detailed evaluation will help in understanding the candidate's fit for the position and guide further decision-making.
    """

    if submit1:
        if uploaded_file is not None and input_text:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(pdf_content, input_text, input_prompt1)
            st.subheader("The Response is:")
            st.write(response)
        else:
            st.write("Please upload the resume and enter the job description.")

    elif submit3:
        if uploaded_file is not None and input_text:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(pdf_content, input_text, input_prompt3)
            st.subheader("The Response is:")
            st.write(response)
        else:
            st.write("Please upload the resume and enter the job description.")
