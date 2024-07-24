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
import time

# Load environment variables
load_dotenv()

# Retrieve API keys from environment variables
groq_api_key = os.getenv("Groq_API_Key")
hf_api_key = os.getenv('HuggingFace_API_Key')

# Debug print statements
print(f"Groq API Key: {groq_api_key}")
print(f"HuggingFace API Key: {hf_api_key}")

# Set HuggingFace API key in environment
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_api_key

# Streamlit application title
st.title("Help/Support Center")

# Initialize the LLM
try:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
except Exception as e:
    st.write(f"Error initializing LLM: {e}")
    st.stop()

# Define the prompt template
prompt_template = """
I want you to act as a "Condition Monitoring Engineer".
Please provide the answer on the basis of the context only.
Please provide the most accurate and precise answer based on the context.

<context>
{context}
<context>
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# Initialize the HuggingFace embeddings
try:
    huggingface_embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
except Exception as e:
    st.write(f"Error initializing HuggingFace embeddings: {e}")
    st.stop()

# Function to initialize vectors and embeddings
def vectors_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = huggingface_embedding
        st.session_state.loader = PyPDFDirectoryLoader('data/')
        try:
            st.session_state.docs = st.session_state.loader.load()
        except Exception as e:
            st.write(f"Error loading documents: {e}")
            st.stop()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        try:
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        except Exception as e:
            st.write(f"Error creating FAISS vectors: {e}")
            st.stop()

# User input for the question
prompt1 = st.text_input("How may I help you:")

# Button to start the embedding process
if st.button("Click here to start"):
    vectors_embedding()
    st.write("Ready to proceed further--")

# Processing the user question
if prompt1:
    if "vectors" in st.session_state:
        try:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            response_time = time.process_time() - start
            
            st.write(f"Response time: {response_time} seconds")
            st.write(response['answer'])

            # Display document similarity search results
            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
        except Exception as e:
            st.write(f"Error processing user question: {e}")
    else:
        st.write("Please click the button to start the embedding process first.")
