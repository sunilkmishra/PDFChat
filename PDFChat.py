import os
import tempfile
from typing import Set
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader  # Updated import
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_openai import OpenAIEmbeddings  # Updated import for OpenAIEmbeddings
from streamlit_chat import message
from langchain_openai import ChatOpenAI  # Updated import for ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from typing import Any, Dict, List
from langchain.prompts import PromptTemplate  # Correct import for PromptTemplate from langchain
from langchain.text_splitter import CharacterTextSplitter  # Correct import for CharacterTextSplitter

# Ensure OpenAI API key is set
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")
os.environ["OPENAI_API_KEY"] = api_key

# Function to create a sources string for output
def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "Sources:\n\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i + 1}.{source}\n\n"
    return sources_string

# Function to run the LLM for querying
def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    docsearch = FAISS.load_local(r"C:\Teaching\GenAI\Excercises\PDFChat", embeddings, allow_dangerous_deserialization=True)
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )
    return qa.invoke({"question": query, "chat_history": chat_history})

# Streamlit user interface for PDF file upload and chat
st.title("PDF Chat Application")
st.header("Upload PDF Files to Build VectorStore")

# Upload multiple PDFs
pdf_files = st.file_uploader("Select PDF files to upload", type="pdf", accept_multiple_files=True)

if pdf_files and st.button('Build Vector Store'):
    # Process uploaded PDFs
    documents = []
    
    for uploaded_file in pdf_files:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load the PDF file from the temporary file path
        loader = PyPDFLoader(tmp_file_path)
        documents.extend(loader.load())
        
        # Optionally, clean up the temporary file after loading
        os.remove(tmp_file_path)

    # Ensure documents are loaded
    if not documents:
        st.error("No documents loaded. Please check the files.")
    else:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        docs = text_splitter.split_documents(documents=documents)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(r"C:\Teaching\GenAI\Excercises\PDFChat")

        st.success("Vector Store Created Successfully!")

# Chatbot interface
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_answer_history" not in st.session_state:
    st.session_state["chat_answer_history"] = []

st.subheader("Ask questions based on the documents")

prompt = st.text_input("Prompt", placeholder="Enter your question here...")
if prompt:
    if st.button('Submit'):
        with st.spinner("Generating responses..."):
            # Create the prompt template with the placeholder for the question
            prompt_template = PromptTemplate.from_template("You are an expert in Generative AI. Your job is to provide accurate answers based on context documents provided. Question: {Question}")
            
            # Format the user question into the template
            New_prompt = prompt_template.format(Question=prompt)
            
            # Generate the response by passing the formatted prompt to the LLM
            generated_response = run_llm(query=New_prompt)
            print(generated_response)
            
            # Get the sources and format the response
            sources = set([doc.metadata["source"] for doc in generated_response["source_documents"]])
            formatted_response = f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
            
            st.write(":red[Disclaimer: The below response is automatically generated using Generative AI, user discretion advised.]")
            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answer_history"].append(formatted_response)

# Display chat history with unique keys for each message
if st.session_state["chat_answer_history"]:
    for idx, (generated_response, user_query) in enumerate(zip(st.session_state["chat_answer_history"], st.session_state["user_prompt_history"])):
        # Display user message with a unique key
        message(user_query, is_user=True, key=f"user_{idx}")
        # Display assistant's response with a unique key
        message(generated_response, key=f"assistant_{idx}")
