# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import fitz  # PyMuPDF for PDF processing
import camelot  # For table extraction from PDFs
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceEndpoint
from langchain.embeddings import HuggingFaceEmbeddings
import os
from pytesseract import pytesseract  # For OCR (optional, if needed for diagrams)

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

# Function to process PDF files with layout information
def process_pdf_with_layout(file):
    pdf_documents = []
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as pdf:
            for page in pdf:
                blocks = page.get_text("dict")["blocks"]
                page_content = []
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            line_text = " ".join([span["text"] for span in line["spans"]])
                            page_content.append(line_text)
                pdf_documents.append(Document(page_content="\n".join(page_content)))
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
    return pdf_documents

# Function to extract tables from PDFs
def extract_tables(file):
    tables = []
    try:
        temp_path = f"temp_{file.name}"  # Temporary save for camelot
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        extracted_tables = camelot.read_pdf(temp_path, pages="all", flavor="stream")
        tables = [table.df for table in extracted_tables]
        os.remove(temp_path)  # Clean up temporary file
    except Exception as e:
        st.error(f"Error extracting tables from PDF: {e}")
    return tables

# Function to normalize text
def normalize_text(text):
    return " ".join(text.split())

# Function to remove duplicates and align data
def remove_duplicates_and_align(data):
    seen = set()
    aligned_data = []
    for line in data.splitlines():
        if line.strip() and line not in seen:
            seen.add(line)
            aligned_data.append(line.strip())
    return "\n".join(aligned_data)

# Function to process Excel files
def process_excel(file):
    try:
        import openpyxl
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
            documents = process_pdf_with_layout(uploaded_file)
            tables = extract_tables(uploaded_file)
            all_documents.extend(documents)
            all_documents.extend([Document(page_content=table.to_string()) for table in tables])
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
    st.title("INSIGHTOR!!!")
    st.subheader("Your Intelligent File Assistant")
    st.write("Upload PDFs, Excel, or CSV files and CHAT with your documents.")

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
        You are an intelligent assistant that extracts and organizes insights from structured and unstructured documents. 
        When responding:
        1. Organize the information into a clear and logical structure (e.g., bullet points, tables).
        2. Avoid repeating or overlapping details. Consolidate similar schemes or categories into distinct sections.
        3. If the data appears unstructured (e.g., scattered, in boxes, or diagrams), interpret and align it logically based on context.
        4. Use only the provided data for answers. If the information is unavailable or unclear, respond with "I don't know."

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
