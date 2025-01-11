# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import pdfplumber  # For PDF text and table extraction
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

# Process PDF using pdfplumber
def process_pdf_with_pdfplumber(pdf_file):
    all_text = []
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                # Extract text
                text = page.extract_text()
                if text:
                    all_text.append(Document(page_content=text))

                # Extract tables
                tables = page.extract_tables()
                for table in tables:
                    table_text = "\n".join([
                        "\t".join(str(cell) if cell is not None else "" for cell in row) for row in table if row
                    ])
                    all_text.append(Document(page_content=table_text))
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
    return all_text

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

    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            extracted_data = process_pdf_with_pdfplumber(file)
            all_documents.extend(extracted_data)
        elif file.name.endswith(".xlsx"):
            all_documents.extend(process_excel(file))
        elif file.name.endswith(".csv"):
            all_documents.extend(process_csv(file))
        else:
            st.warning(f"Unsupported file type: {file.name}")

    return all_documents

# Create vector store
def create_vector_store(documents):
    try:
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_documents(documents, embedding)
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# Clean the LLM's response
def clean_response(response):
    response = response.split("Thank you!")[0]  # Remove verbose endings
    cleaned_lines = [line.strip() for line in response.split("\n") if line.strip()]
    return "\n".join(cleaned_lines)

# Generate response
def generate_response(query, qa):
    try:
        result = qa({"query": query})
        cleaned_result = clean_response(result["result"])
        return cleaned_result
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

        # Define prompt for structured and unstructured data
        prompt_template = """
        You are an intelligent assistant tasked with extracting precise insights from structured and unstructured documents. 

        When answering:
        1. Provide a concise and factual summary of the relevant details.
        2. Clearly distinguish between different schemes or sections, and avoid merging or repeating details.
        3. Use bullet points or numbers for clarity.
        4. Avoid unnecessary phrases like "I don't know about any other schemes" or "Thank you!"

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
