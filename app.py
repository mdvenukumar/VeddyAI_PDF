import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

# Set API Key
os.environ['GOOGLE_API_KEY'] = st.secrets["API_KEY"]

# Function to get PDF text
def get_pdf_text(pdf_docs):
    try:
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

# Function to get text chunks
def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text into chunks: {e}")
        return None

# Function to get vector store
def get_vector_store(text_chunks):
    try:
        embeddings = GooglePalmEmbeddings()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# Function to get conversational chain
def get_conversational_chain(vector_store):
    try:
        llm = GooglePalm()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversational chain: {e}")
        return None

# Function for user input
def user_input(user_question):
    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chatHistory = response['chat_history']
        for i, message in enumerate(st.session_state.chatHistory):
            if i % 2 == 0:
                st.write("Human: ", message.content)
            else:
                st.write("Bot: ", message.content)
    except Exception as e:
        st.error(f"Error processing user input: {e}")

# Main function
def main():
    st.set_page_config("Veddy AI")
    st.header("Chat with your PDF 💬")

    # Instructions for file upload
    st.subheader("Upload your PDF Files")

    # Sidebar settings
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Process Button", accept_multiple_files=True)

    if pdf_docs:
        with st.spinner("Uploading..."):
            # Process uploaded files
            raw_text = get_pdf_text(pdf_docs)
            if raw_text:
                text_chunks = get_text_chunks(raw_text)
                if text_chunks:
                    vector_store = get_vector_store(text_chunks)
                    if vector_store:
                        st.session_state.conversation = get_conversational_chain(vector_store)
                        st.success("Processing completed successfully.")
                        st.subheader("Ask a Question from the PDF Files")
                        user_question = st.text_input("Type your question here")
                        if user_question:
                            user_input(user_question)
                    else:
                        st.warning("Error creating vector store. Please try again.")
                else:
                    st.warning("Error splitting text into chunks. Please try again.")
            else:
                st.warning("Error extracting text from PDF. Please check the uploaded files.")
    else:
        st.info("Please upload PDF files.")

    # Hide Streamlit toolbar and add a custom footer
    hide_streamlit_style = """
        <style>
        [data-testid="stToolbar"] {visibility: hidden !important;}
        footer {visibility: hidden !important;}
        </style>
        """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.markdown(
        """
        <div style="position: fixed; bottom: 10px; left: 10px; background-color: #ff4b4b; padding: 10px; border-radius: 8px; color: white;">
            Thevk22
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
