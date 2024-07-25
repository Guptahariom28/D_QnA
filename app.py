import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

from dotenv import load_dotenv
load_dotenv()

os.environ['HUGGINGFACEHUB_API_TOKEN']= os.getenv("HuggingFace_API_Key")
groq_api_key= os.getenv("Groq_API_Key")

st.title("Hello!")

llm= ChatGroq(groq_api_key= groq_api_key,
              model_name= "Llama-3.1-8b-Instant")

huggingface_embedding= HuggingFaceEmbeddings(
     model_name= ("sentence-transformers/all-mpnet-base-v2"),
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}
)

prompt= ChatPromptTemplate.from_template(
    """
    I want you to act as a "Condition Monitoring Engineer".
    please provide the ansert on the basis of the context only.
    The answer should be accurate and from the context.

    <context>
    {context}
    <context>
    Question: {question}
    """
)

def vector_embedding():
    if 'vectors' not in st.session_state:
        st.session_state.embeddings=  huggingface_embedding
        st.session_state.loader= PyPDFDirectoryLoader("data\\")
        st.session_state.docs= st.session_state.loader.load()
        st.session_state.text_splitter= RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap=200)
        st.session_state.final_doc= st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors= FAISS.from_documents(st.session_state.final_docs, st.session_state.embedding)

if st.button("Click here to start"):
    vector_embedding()
    st.write("Ready to answer")

prompt1= st.text_input("Enter the Question from the Document")


if prompt1:
    
    document_chain= create_stuff_documents_chain(llm, prompt)
    retriever= st.session_state.vectors.as_retriever()
    retrieval_chain= create_retrieval_chain(retriever, document_chain)
    responde= retrieval_chain(invoke('input', prompt1))
    st.write(rresponse['answer'])
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
