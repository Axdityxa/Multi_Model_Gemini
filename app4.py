import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.llms.base import BaseLanguageModel

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("API key for Google Generative AI is not set. Please check your environment variables.")
    st.stop()

try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Failed to configure Google Generative AI: {e}")
    st.stop()

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Failed to create and save vector store: {e}")
        st.stop()

# Wrapper class to ensure compatibility with BaseLanguageModel
class GoogleGenerativeAIWrapper(BaseLanguageModel):
    def __init__(self, model_name, temperature):
        self._model = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

    def generate_prompt(self, *args, **kwargs):
        return self._model.generate_prompt(*args, **kwargs)

    def agenerate_prompt(self, *args, **kwargs):
        return self._model.agenerate_prompt(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self._model.predict(*args, **kwargs)

    def apredict(self, *args, **kwargs):
        return self._model.apredict(*args, **kwargs)

    def predict_messages(self, *args, **kwargs):
        return self._model.predict_messages(*args, **kwargs)

    def apredict_messages(self, *args, **kwargs):
        return self._model.apredict_messages(*args, **kwargs)

# Function to create a conversational chain
def get_conversational_chain():
    prompt_template = """
    You are tasked with providing comprehensive and accurate answers based on the provided context. Your response should include all relevant details and specifics found within the context. If the necessary information is not present, simply state, "The answer is not available in the context," without making any assumptions or providing incorrect information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    try:
        # Use the wrapper class to ensure compatibility
        model = GoogleGenerativeAIWrapper(model_name="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = LLMChain(llm=model, prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Failed to create conversational chain: {e}")
        st.stop()

# Function to handle user input and get a response
def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings)
        docs = new_db.similarity_search(user_question)
        
        # Prepare context from documents
        context = " ".join([doc.page_content for doc in docs])
        
        # Create chain and get response
        chain = get_conversational_chain()
        response = chain.run({"context": context, "question": user_question})
        
        # Extract and display raw response
        st.write(response)
    except Exception as e:
        st.error(f"Failed to process user input and generate response: {e}")

# Main function for app4
def app4_main():
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Processing complete")
                    except Exception as e:
                        st.error(f"An error occurred during processing: {e}")
            else:
                st.error("Please upload at least one PDF file.")

if __name__ == "__main__":
    app4_main()
