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
from stqdm import stqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize models once (cached)
@st.cache_resource
def load_models():
    return {
        "embedding": HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        "sentence_model": SentenceTransformer('all-MiniLM-L6-v2')
    }

# Improved LLM initialization with fallback
def initialize_llm():
    models_to_try = [
        "mistralai/Mistral-7B-Instruct-v0.1",  # Primary fallback
        "google/gemma-7b-it",                  # Secondary fallback
        "meta-llama/Llama-2-7b-chat-hf"        # Only if access granted
    ]
    
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    if not HUGGINGFACE_TOKEN:
        st.warning("HUGGINGFACE_TOKEN not set. Using limited capabilities.")
        return None

    for repo_id in models_to_try:
        try:
            llm = HuggingFaceEndpoint(
                repo_id=repo_id,
                huggingfacehub_api_token=HUGGINGFACE_TOKEN,
                max_length=1024,
                temperature=0.3
            )
            # Test connection
            llm("Test")
            st.success(f"Connected to {repo_id}")
            return llm
        except Exception as e:
            st.warning(f"Failed to initialize {repo_id}: {str(e)[:100]}...")
            continue
    
    st.error("All LLM attempts failed. Check token and model access.")
    return None

# Robust PDF processing with multiple fallbacks
def process_pdf_with_pdfplumber(pdf_file):
    all_text = []
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                # Attempt 1: Normal extraction
                text = page.extract_text()
                if not text:
                    # Attempt 2: Loose extraction
                    text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if text:
                    all_text.append(Document(page_content=text.strip()))

                # Table extraction with error handling
                try:
                    tables = page.extract_tables()
                    for table in tables:
                        if table and any(any(cell for cell in row) for row in table):
                            table_text = "\n".join([
                                "\t".join(str(cell) if cell is not None else "" 
                                for row in table if row 
                                for cell in row
                            ])
                            all_text.append(Document(page_content=table_text))
                except Exception as table_error:
                    st.warning(f"Table extraction failed on page {page.page_number}: {table_error}")
    except Exception as e:
        st.error(f"PDF processing error: {str(e)[:200]}")
    return all_text

# Enhanced file processing
def process_files(uploaded_files):
    all_documents = []
    for file in uploaded_files:
        try:
            if file.name.endswith(".pdf"):
                extracted_data = process_pdf_with_pdfplumber(file)
                all_documents.extend(extracted_data)
            elif file.name.endswith(".xlsx"):
                df = pd.read_excel(file)
                documents = [
                    Document(page_content=" ".join(map(str, row.values)))
                    for _, row in df.iterrows()
                ]
                all_documents.extend(documents)
            elif file.name.endswith(".csv"):
                df = pd.read_csv(file)
                documents = [
                    Document(page_content=" ".join(map(str, row.values)))
                    for _, row in df.iterrows()
                ]
                all_documents.extend(documents)
            else:
                st.warning(f"Skipped unsupported file: {file.name}")
        except Exception as e:
            st.error(f"Failed to process {file.name}: {str(e)[:100]}...")
    
    return all_documents

# Smart content processing with chunking
def preprocess_content(documents, max_chars=5000):
    if not documents:
        return ""
    
    # Combine and clean content
    raw_content = "\n".join(doc.page_content.strip() 
                           for doc in documents 
                           if doc.page_content.strip())
    
    # Remove redundant content
    models = load_models()
    sentences = [s for s in raw_content.split(". ") if s]
    if len(sentences) > 1:
        embeddings = models["sentence_model"].encode(sentences)
        unique_indices = []
        for i in range(len(sentences)):
            if not any(cosine_similarity([embeddings[i]], [embeddings[j]])[0,0] > 0.8 
                   for j in unique_indices):
                unique_indices.append(i)
        filtered_content = ". ".join(sentences[i] for i in unique_indices)
    else:
        filtered_content = raw_content
    
    # Intelligent chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(filtered_content[:max_chars*3])  # Buffer for splitting
    
    return "\n\n".join(chunks[:3])  # Return top 3 most relevant chunks

# Analytics generation with enhanced prompts
def generate_analytics(documents, llm):
    if not llm:
        st.error("LLM not available for analytics")
        return
    
    content = preprocess_content(documents)
    if not content:
        st.error("No usable content found")
        return

    prompt_template = """Generate a comprehensive analytics report with these sections:
    1. **Key Trends** (3-5 bullet points)
    2. **Statistical Highlights** (top 5 numerical insights)
    3. **Anomalies Detected** (unexpected patterns)
    4. **Actionable Recommendations** (prioritized list)

    Content: {content}
    """
    
    try:
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["content"]
        )
        response = llm(prompt.format(content=content[:5000]))
        
        st.subheader("Analytics Report")
        st.markdown(response)
        
        # Visualizations
        st.subheader("Data Visualizations")
        visualize_content(content)
        
    except Exception as e:
        st.error(f"Analytics generation failed: {str(e)[:200]}")

# Visualization utilities
def visualize_content(content):
    # Word cloud
    try:
        wordcloud = WordCloud(width=800, height=400).generate(content)
        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud)
        plt.axis("off")
        st.pyplot(plt)
    except:
        st.warning("Could not generate word cloud")

    # Term frequency
    try:
        vectorizer = CountVectorizer(max_features=10, stop_words='english')
        X = vectorizer.fit_transform([content])
        terms = vectorizer.get_feature_names_out()
        freqs = X.toarray()[0]
        df = pd.DataFrame({'Term': terms, 'Frequency': freqs})
        df = df.sort_values('Frequency', ascending=False)
        
        fig, ax = plt.subplots()
        sns.barplot(data=df, x='Frequency', y='Term', ax=ax)
        st.pyplot(fig)
    except:
        st.warning("Could not generate term frequencies")

# Main application flow
def main():
    st.set_page_config(page_title="INSIGHTOR 2.0", layout="wide")
    st.title("📊 INSIGHTOR 2.0")
    st.markdown("### Your Robust Document Analysis Platform")

    # Model initialization
    models = load_models()
    llm = initialize_llm()

    # File upload
    uploaded_files = st.file_uploader(
        "Upload documents (PDF/Excel/CSV)",
        type=["pdf", "xlsx", "csv"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Upload files to begin analysis")
        return

    # Processing with progress
    with st.spinner("Processing files..."):
        documents = process_files(uploaded_files)
        if not documents:
            st.error("No valid content extracted")
            return
        st.success(f"Processed {len(documents)} document chunks")

    # Analysis options
    option = st.radio(
        "Select analysis mode:",
        ("Analytics Report", "Document Chat", "Data Visualization"),
        horizontal=True
    )

    if option == "Analytics Report":
        generate_analytics(documents, llm)
    elif option == "Document Chat" and llm:
        setup_chat_interface(documents, llm, models["embedding"])
    elif option == "Data Visualization":
        content = preprocess_content(documents)
        visualize_content(content)

def setup_chat_interface(documents, llm, embeddings):
    st.subheader("Document Q&A")
    
    # Create vector store
    try:
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Vector store failed: {str(e)[:200]}")
        return

    # Chat memory
    if "chat_memory" not in st.session_state:
        st.session_state.chat_memory = ConversationBufferMemory()

    # Chat chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=st.session_state.chat_memory,
        verbose=True
    )

    # Chat interface
    query = st.text_input("Ask about your documents:")
    if query:
        try:
            result = qa_chain({"query": query})
            st.markdown(f"**Answer:** {result['result']}")
            
            # Show context
            with st.expander("See sources"):
                for doc in result.get('source_documents', []):
                    st.text(doc.page_content[:200] + "...")
        except Exception as e:
            st.error(f"Query failed: {str(e)[:200]}")

if __name__ == "__main__":
    main()
