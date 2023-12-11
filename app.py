import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

# Checking if the API key is set
if "API_KEY" not in st.secrets:
    st.warning("Please set your Google API key in Streamlit secrets.")
    st.stop()

os.environ['GOOGLE_API_KEY'] = st.secrets["API_KEY"]

# Improved function to get text from PDFs with error handling
def get_pdf_text(pdf_docs):
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return text

# Improved function to get text chunks with error handling
def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text into chunks: {e}")
        return []

# Improved function to get vector store with error handling
def get_vector_store(text_chunks):
    try:
        embeddings = GooglePalmEmbeddings()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# Improved function to get conversational chain with error handling
def get_conversational_chain(vector_store):
    try:
        llm = GooglePalm()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversational chain: {e}")
        return None

# Improved function for user input with error handling and input reset
def user_input(user_question):
    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chatHistory = response['chat_history']
        for i, message in enumerate(st.session_state.chatHistory):
            if i % 2 == 0:
                st.write("Human: ", message.content)
            else:
                st.write("Bot: ", message.content)
        
        # Clear the input field after processing the user question
        st.text_input("Ask a Question from the PDF Files", value="", key="user_input")
    except Exception as e:
        st.error(f"Error processing user input: {e}")

# Main function with enhanced UI and error handling
def main():
    st.set_page_config("Veddy AI", layout="wide")
    
    # Improved UI for the header
    st.title("Chat with your PDF 💬")
    st.markdown("---")
    
    # Improved UI for user input
    user_question = st.text_input("Ask a Question from the PDF Files", key="user_input")
    
    # Initializing session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    
    # Processing user input
    if user_question:
        user_input(user_question)
    
    # Improved UI for sidebar settings
    with st.sidebar:
        st.title("Settings")
        st.subheader("Upload your Documents")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        
        # Processing PDFs and building the conversational chain
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    
                    # Error handling for vector store creation
                    vector_store = get_vector_store(text_chunks)
                    if vector_store is not None:
                        st.session_state.conversation = get_conversational_chain(vector_store)
                        st.success("Processing complete")
                    else:
                        st.warning("Processing failed. Please check the logs for details.")
            else:
                st.warning("Please upload PDF files before processing.")
    
    # Hiding Streamlit toolbar and footer
    hide_streamlit_style = """
        <style>
        [data-testid="stToolbar"] {visibility: hidden !important;}
        footer {visibility: hidden !important;}
        </style>
        """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Adding a signature with improved styling
    st.markdown(
        """
        <div style="position: fixed; bottom: 10px; left: 10px; background-color: #ff4b4b; padding: 10px; border-radius: 8px; color: white; font-size: 14px;">
            Powered by Veddy AI
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
