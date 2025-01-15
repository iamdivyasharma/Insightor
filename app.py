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
import os
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy

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

# Process text from PDFs
def process_pdf_text(pdf_file):
    text_data = []
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_data.append(Document(page_content=text))
    except Exception as e:
        st.error(f"Error processing PDF text: {e}")
    return text_data

# Sentiment Analysis
def sentiment_analysis(documents):
    st.write("### Sentiment Analysis")
    from textblob import TextBlob

    sentiments = []
    for doc in documents:
        analysis = TextBlob(doc.page_content)
        sentiments.append(analysis.sentiment.polarity)

    fig, ax = plt.subplots()
    sns.histplot(sentiments, kde=True, ax=ax)
    ax.set_title("Sentiment Polarity Distribution")
    ax.set_xlabel("Polarity")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.write("**Insights:**")
    st.write("- Polarity ranges from -1 (negative) to 1 (positive).")
    st.write(f"- Average Sentiment Polarity: {sum(sentiments) / len(sentiments):.2f}")

# Named Entity Recognition
def named_entity_recognition(documents):
    st.write("### Named Entity Recognition")
    nlp = spacy.load("en_core_web_sm")
    entity_counts = {}

    for doc in documents:
        spacy_doc = nlp(doc.page_content)
        for ent in spacy_doc.ents:
            entity_counts[ent.label_] = entity_counts.get(ent.label_, 0) + 1

    labels, counts = zip(*entity_counts.items())

    fig, ax = plt.subplots()
    sns.barplot(x=list(labels), y=list(counts), ax=ax)
    ax.set_title("Entity Counts by Type")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Entity Type")
    st.pyplot(fig)

# Topic Modeling
def topic_modeling(documents):
    st.write("### Topic Modeling")

    # Prepare data
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
    all_text = [doc.page_content for doc in documents]
    dtm = vectorizer.fit_transform(all_text)

    # Apply LDA
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(dtm)

    topics = lda.components_
    feature_names = vectorizer.get_feature_names_out()

    st.write("**Top Words Per Topic:**")
    for topic_idx, topic in enumerate(topics):
        st.write(f"Topic {topic_idx + 1}: {', '.join([feature_names[i] for i in topic.argsort()[:-6:-1]])}")

    # Visualize topics
    fig, ax = plt.subplots()
    sns.heatmap(lda.components_, cmap="YlGnBu", yticklabels=[f"Topic {i + 1}" for i in range(len(topics))],
                xticklabels=False, cbar_kws={"label": "Word Importance"})
    ax.set_title("Topic Heatmap")
    st.pyplot(fig)

# Unified file processing
def process_files(uploaded_files):
    all_documents = []
    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            text_data = process_pdf_text(file)
            all_documents.extend(text_data)
        elif file.name.endswith(".xlsx"):
            all_documents.extend(process_excel(file))
        elif file.name.endswith(".csv"):
            all_documents.extend(process_csv(file))
        else:
            st.warning(f"Unsupported file type: {file.name}")
    return all_documents

# Allow direct text input
def direct_text_input():
    st.write("### Direct Text Input")
    text_input = st.text_area("Enter your text here:", "")
    if text_input:
        return [Document(page_content=text_input)]
    return []

# Main Streamlit app
def main():
    st.title("INSIGHTOR 2.0")
    st.subheader("Your Unified Document Processing Platform")

    st.write("### Data Ingestion")
    ingestion_option = st.radio(
        "Choose your data ingestion method:",
        ("Upload Files", "Direct Text Input")
    )

    documents = []

    if ingestion_option == "Upload Files":
        uploaded_files = st.file_uploader(
            "Upload your files (PDF, Excel, CSV, Images)", 
            type=["pdf", "xlsx", "csv", "png", "jpg"], 
            accept_multiple_files=True
        )
        if uploaded_files:
            documents = process_files(uploaded_files)
            if not documents:
                st.warning("No valid documents found. Please upload supported files.")
                return
    elif ingestion_option == "Direct Text Input":
        documents = direct_text_input()
        if not documents:
            st.warning("No text entered. Please provide input.")
            return

    if not documents:
        return

    vectorstore = create_vector_store(documents)
    if vectorstore is None:
        return

    llm = initialize_llm()

    st.write("### Select an Option:")
    option = st.radio(
        "What would you like to do?",
        ("Analytics Report", "Sentiment Analysis", "NER", "Topic Modeling", "Chat with Your Document")
    )

    if option == "Analytics Report":
        generate_advanced_analytics(documents)

    elif option == "Sentiment Analysis":
        sentiment_analysis(documents)

    elif option == "NER":
        named_entity_recognition(documents)

    elif option == "Topic Modeling":
        topic_modeling(documents)

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

if __name__ == "__main__":
    main()
