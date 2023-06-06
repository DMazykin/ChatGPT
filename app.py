from chromadb.config import Settings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma  # for the vectorization part
from langchain.llms import OpenAI  # the LLM model we'll use (CHatGPT)
from langchain.chains import RetrievalQA
import streamlit as st
import chromadb
import os

embeddings = OpenAIEmbeddings()

client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="db/"
    )
)

st.title('Ask questions about your PDF')

docs = [collection.name for collection in client.list_collections()]

document = st.selectbox("Choose document", options=docs)

if document:

    retriever = Chroma(embedding_function=embeddings,
                       persist_directory="db/",
                       collection_name=document
                       ).as_retriever()

    QA = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="map_reduce",#"stuff",
        retriever=retriever
    )

    # Create a text input box for the user
    question = st.text_input('Ask your question here')

    # If the user hits enter
    if question:
        with st.spinner("Generating answer"):
            st.write(QA.run(question))
