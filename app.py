# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import fitz  # PyMuPDF for PDF processing
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceEndpoint
from langchain.embeddings import HuggingFaceEmbeddings
import os

# Function to initialize the LLM
def initialize_llm():
    REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  # Replace with your model
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # Load token from environment variable
    if not HUGGINGFACE_TOKEN:
        raise ValueError("HUGGINGFACE_TOKEN environment variable is not set.")
    return HuggingFaceEndpoint(
        repo_id=REPO_ID,
        huggingfacehub_api_token=HUGGINGFACE_TOKEN,
        max_length=512,
        temperature=0.5
    )

# Function to process PDF files
def process_pdf(file):
    pdf_documents = []
    try:
        with fitz.open(stream=file) as pdf:
            for page in pdf:
                text = page.get_text("text")
                if text.strip():  # Ignore empty pages
                    pdf_documents.append(Document(page_content=text))
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
    return pdf_documents

# Function to process Excel files
def process_excel(file):
    try:
        df = pd.read_excel(file)
        documents = [
            Document(page_content=" ".join(map(str, row.values)))
            for _, row in df.iterrows()
        ]
        return documents
    except Exception as e:
        st.error(f"Error processing Excel file: {e}")
        return []

# Function to process CSV files
def process_csv(file):
    try:
        df = pd.read_csv(file)
        documents = [
            Document(page_content=" ".join(map(str, row.values)))
            for _, row in df.iterrows()
        ]
        return documents
    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        return []

# Function to handle all file types
def process_files(uploaded_files):
    all_documents = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".pdf"):
            all_documents.extend(process_pdf(uploaded_file))
        elif uploaded_file.name.endswith(".xlsx"):
            all_documents.extend(process_excel(uploaded_file))
        elif uploaded_file.name.endswith(".csv"):
            all_documents.extend(process_csv(uploaded_file))
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
    return all_documents

# Create vector store
def create_vector_store(documents):
    try:
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_documents(documents, embedding)
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# Generate response
def generate_response(query, qa):
    try:
        result = qa({"query": query})
        return result["result"]
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, something went wrong."

# Main Streamlit app
def main():
    st.title("Enhanced RAG Chatbot for Unstructured Data")
    st.write("Upload PDFs, Excel, or CSV files to get fast and accurate insights.")

    # File upload
    uploaded_files = st.file_uploader(
        "Upload your files (PDF, Excel, or CSV)", 
        type=["pdf", "xlsx", "csv"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        documents = process_files(uploaded_files)
        if not documents:
            st.warning("No valid documents found. Please upload supported files.")
            return

        vectorstore = create_vector_store(documents)
        if vectorstore is None:
            return  # Stop if vectorstore creation fails

        llm = initialize_llm()

        # Define prompt for LLM
        prompt_template = """
        You are an intelligent assistant providing insights from the given context.
        Use only the provided data. If information is unavailable, reply with "I don't know."

        Context: {context}
        Question: {question}
        """
        llama_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # Create RetrievalQA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": llama_prompt},
            return_source_documents=False
        )

        # Query input
        query = st.text_input("Ask your question:")
        if query:
            response = generate_response(query, qa)
            st.write(f"**Bot:** {response}")

if __name__ == "__main__":
    main()
