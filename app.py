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
from pdf2image import convert_from_bytes
from pytesseract import pytesseract, Output
from PIL import Image, ImageDraw
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

# Convert PDF pages to images
def convert_pdf_to_images(pdf_file):
    images = convert_from_bytes(pdf_file.read())
    return images

# Detect text and bounding boxes using OCR
def detect_text_with_boxes(image):
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    boxes = []
    for i in range(len(data["level"])):
        (x, y, w, h) = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
        text = data["text"][i].strip()
        if text:
            boxes.append({"text": text, "bbox": (x, y, w, h)})
    return boxes

# Mark detected text boxes on the image
def mark_text_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        (x, y, w, h) = box["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
    return image

# Align and group detected text logically
def align_and_group_boxes(boxes):
    sorted_boxes = sorted(boxes, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    grouped_data = []
    for box in sorted_boxes:
        grouped_data.append(box["text"])
    return "\n".join(grouped_data)

# Process PDF pages with images and OCR
def process_pdf_page(pdf_file):
    images = convert_pdf_to_images(pdf_file)
    all_extracted_data = []

    for image in images:
        # Detect and mark text boxes
        boxes = detect_text_with_boxes(image)
        marked_image = mark_text_boxes(image.copy(), boxes)
        marked_image.show()  # Optional: Display the image with bounding boxes

        # Align and group text
        text_data = align_and_group_boxes(boxes)
        all_extracted_data.append(text_data)

    # Extract structured tables using Camelot
    try:
        temp_path = f"temp_{pdf_file.name}"  # Temporary save for camelot
        with open(temp_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        tables = camelot.read_pdf(temp_path, pages="all", flavor="stream")
        for table in tables:
            all_extracted_data.append(table.df.to_string())
        os.remove(temp_path)  # Clean up temporary file
    except Exception as e:
        st.warning(f"Error processing tables: {e}")

    return all_extracted_data

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
            extracted_data = process_pdf_page(file)
            all_documents.extend([Document(page_content=data) for data in extracted_data])
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
