import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv

load_dotenv()

groq_api_key= os.getenv("Groq_API_Key")
os.environ['HUGGINGFACEHUB_API_TOKEN']= os.getenv('HuggingFace_API_Key')

st.title("Help/Support  Center")

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-8b-8192")

prompt= ChatPromptTemplate.from_template("""
I want you to act as a "Condition Monitoring Engineer".
Please prpvide the answer on the bsis on the context only.
Please provide most accurate and precise answer based on the context.

<context>
{context}
<context>
Question:{question}

"""
)

huggingface_embedding= HuggingFaceEmbeddings(
    model_name= ("BAAI/bge-m3"),
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}
)

def vectors_embedding():

    if "vectors" not in st.session_state:

        st.session_state.embeddings= huggingface_embedding
        st.session_state.loader=PyPDFDirectoryLoader('data/')
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

prompt1=st.text_input("How may I help you:")

if st.button("Click here to start"):
    vectors_embedding()
    st.write("Ready to proceed further--")
    
import time

if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")

