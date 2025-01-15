import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pdfplumber
import cv2
import pytesseract
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceEndpoint
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain.llms import OpenAI

# Function to initialize the LLM
def initialize_llm():
    REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
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
    if not documents:
        st.error("No valid documents to create a vector store.")
        return None
    try:
        st.write("Initializing embeddings model...")
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.write("Creating vector store...")
        return FAISS.from_documents(documents, embedding)
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# Remove redundant sentences using semantic similarity
def remove_redundant_sentences(content):
    # Break the content into sentences
    sentences = content.split(". ")
    if not sentences:
        return content

    # Load a pre-trained model for sentence embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)

    # Keep sentences with low redundancy
    unique_sentences = []
    for i, sentence in enumerate(sentences):
        if not any(cosine_similarity([embeddings[i]], [embeddings[j]])[0, 0] > 0.85 for j in range(len(unique_sentences))):
            unique_sentences.append(i)

    return ". ".join([sentences[i] for i in unique_sentences])

# CSV Agent functionality
def create_csv_agent_interface(file):
    """Creates a CSV agent for analyzing uploaded CSV or Excel files."""
    if file.name.endswith("csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    temp_csv_path = "temp_file.csv"
    df.to_csv(temp_csv_path, index=False)

    try:
        agent = create_csv_agent(OpenAI(), temp_csv_path, verbose=True)
        return agent
    except Exception as e:
        st.error(f"Error creating CSV agent: {e}")
        return None

# Generate analytics report using LLM
def generate_analytics(documents, llm):
    st.write("### Analytics Report")
    prompt_template = """
    Generate a detailed analytics report based on the provided documents. The report should address the following key areas:

    1. **Key Trends and Patterns**: Summarize significant trends and recurring patterns in the data.
    2. **Root Cause Analysis**: Identify the underlying reasons for observed trends or anomalies.
    3. **Top Drivers and Detractors**: Highlight factors contributing to positive outcomes and those causing challenges or declines.
    4. **Competitor Analysis**: Compare performance or metrics with top competitors, where applicable.
    5. **Quarter/Year Comparisons**: Provide insights into changes compared to the previous quarter or year.
    6. **Unusual Trends or Anomalies**: Detect and explain unexpected findings or deviations in the data.
    7. **Statistical Highlights**: Summarize key numbers, metrics, or statistical findings.
    8. **Actionable Recommendations**: Provide specific, practical recommendations to improve performance, mitigate risks, and leverage opportunities.

    Documents:
    {documents}
    """
    raw_content = "\n".join(doc.page_content.strip() for doc in documents[:5] if doc.page_content.strip())
    filtered_content = remove_redundant_sentences(raw_content)
    document_text = filtered_content[:3000]  # Limit to 3000 characters

    if not document_text:
        st.error("No valid content available for analytics.")
        return

    prompt = PromptTemplate(template=prompt_template, input_variables=["documents"])
    analytics_prompt = prompt.format(documents=document_text)

    try:
        response = llm(analytics_prompt)
        st.write("### Key Insights")
        st.markdown(response)
    except Exception as e:
        st.error(f"Error generating analytics report: {e}")

# Generate summarization using LLM
def generate_summary(documents, llm):
    st.write("### Summary")
    prompt_template = """
    Summarize the following documents concisely, eliminating repetitive content and focusing on key points:
    Documents:
    {documents}
    """
    raw_content = "\n".join(doc.page_content.strip() for doc in documents[:5] if doc.page_content.strip())
    filtered_content = remove_redundant_sentences(raw_content)
    document_text = filtered_content[:2000]  # Limit length to 2000 characters

    if not document_text:
        st.error("No valid content available for summarization.")
        return

    prompt = PromptTemplate(template=prompt_template, input_variables=["documents"])
    summary_prompt = prompt.format(documents=document_text)

    try:
        response = llm(summary_prompt)
        st.write(response)
    except Exception as e:
        st.error(f"Error generating summary: {e}")

# Generate recommendations using LLM
def generate_recommendations(documents, llm):
    st.write("### Recommendations")
    prompt_template = """
    Based on the content of the following documents, generate actionable recommendations to improve processes or decision-making.
    Analyze the following documents and provide actionable and specific recommendations. Ensure:
    1. Each recommendation is unique and avoids redundancy.
    2. Address key challenges with concrete strategies.
    3. Highlight opportunities for innovation or efficiency improvements.
    4. Provide risk mitigation steps supported by the document's insights.
    5. Recommendations should be categorized where applicable, such as 'Operational', 'Strategic', or 'Risk Management'.
    Documents:
    {documents}
    """
    document_text = "\n".join(doc.page_content[:1000] for doc in documents[:5])
    prompt = PromptTemplate(template=prompt_template, input_variables=["documents"])
    recommendations_prompt = prompt.format(documents=document_text)

    try:
        response = llm(recommendations_prompt)
        st.write(response)
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")

# Main Streamlit app
def main():
    st.title("INSIGHTOR 2.0")
    st.subheader("Your Unified Document Processing Platform")

    # Step 1: User selects an option
    st.write("### Select an Option:")
    option = st.radio(
        "What would you like to do?",
        ("None", "Analytics Report", "Summarization", "Recommendations", "Chat with Your Document", "Visualize Tables and Numbers", "CSV Agent Analytics")
    )

    if option == "None":
        st.info("Please select an option to proceed.")
        return

    # Step 3: File ingestion
    st.write("### Data Ingestion")
    uploaded_files = st.file_uploader(
        "Upload your files (PDF, Excel, CSV, Images)", 
        type=["pdf", "xlsx", "csv", "png", "jpg"], 
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.warning("Please upload at least one file to continue.")
        return

    documents = process_files(uploaded_files)
    if not documents:
        st.error("No valid documents were processed.")
        return
    st.write(f"Processed {len(documents)} documents.")

    if option == "CSV Agent Analytics":
        for file in uploaded_files:
            if file.name.endswith("csv") or file.name.endswith("xlsx"):
                agent = create_csv_agent_interface(file)
                if agent:
                    user_query = st.text_input(f"Ask a question about {file.name}:")
                    if user_query:
                        try:
                            result = agent.run(user_query)
                            st.write("Result:", result)
                        except Exception as e:
                            st.error(f"Error querying the CSV agent: {e}")
                else:
                    st.error("Failed to create CSV agent.")
            else:
                st.warning(f"{file.name} is not a CSV or Excel file.")

    vectorstore = create_vector_store(documents)
    if not vectorstore:
        st.error("Vector store could not be created.")
        return

    try:
        llm = initialize_llm()
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return

    if option == "Analytics Report":
        generate_analytics(documents, llm)
    elif option == "Summarization":
        generate_summary(documents, llm)
    elif option == "Recommendations":
        generate_recommendations(documents, llm)
    elif option == "Chat with Your Document":
        prompt_template = """
        You are an intelligent assistant tasked with extracting precise insights from structured and unstructured documents. 
        Provide concise and factual summaries using the given context.
        Context: {context}
        Question: {question}
        """
        llama_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": llama_prompt},
            return_source_documents=False
        )

        query = st.text_input("Ask your question:")
        if query:
            try:
                result = qa({"query": query})
                st.write(f"**Bot:** {result['result']}")
            except Exception as e:
                st.error(f"Error generating response: {e}")
    elif option == "Visualize Tables and Numbers":
        visualize_tables_and_numbers(documents)

if __name__ == "__main__":
    main()
