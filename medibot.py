# medibot.py

import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

def load_environment():
    """
    Load environment variables from .env file.
    """
    load_dotenv(find_dotenv())
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        st.error("HF_TOKEN not found in environment variables. Please set it in the .env file.")
        st.stop()
    return hf_token

@st.cache_resource
def get_vectorstore(db_path: str, embedding_model_name: str):
    """
    Load the FAISS vector store with the specified embedding model.
    """
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(template: str) -> PromptTemplate:
    """
    Define a custom prompt template.
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    return prompt

def load_llm(huggingface_repo_id: str, hf_token: str) -> HuggingFaceEndpoint:
    """
    Initialize the HuggingFace LLM endpoint.
    """
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        token=hf_token,
        model_kwargs={
            "max_length": 512  # Integer value
        }
    )
    return llm

def main():
    st.set_page_config(page_title="MediBot - Your Medical Assistant", page_icon="💊")
    st.title("💊 MediBot - Your Medical Assistant")

    # Initialize session state for messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        role = message['role']
        content = message['content']
        if role == 'user':
            st.chat_message("user").markdown(content)
        elif role == 'assistant':
            st.chat_message("assistant").markdown(content)

    # User input
    prompt = st.chat_input("Ask your medical question here:")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Define custom prompt template
        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know; don't try to make up an answer. 
        Don't provide anything out of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk, please.
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
        DB_FAISS_PATH = "vectorstore/db_faiss"

        try:
            # Load environment variables
            hf_token = load_environment()

            # Load vector store
            vectorstore = get_vectorstore(DB_FAISS_PATH, EMBEDDING_MODEL_NAME)

            # Initialize LLM
            llm = load_llm(HUGGINGFACE_REPO_ID, hf_token)

            # Initialize RetrievalQA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            # Invoke the chain
            response = qa_chain.invoke({'question': prompt})

            # Extract result and source documents
            result = response.get("result", "No answer found.")
            source_documents = response.get("source_documents", [])

            # Format source documents
            if source_documents:
                sources = "\n".join(
                    [f"- {doc.metadata.get('source', 'Unknown Source')}" for doc in source_documents]
                )
                result_to_show = f"{result}\n\n**Source Documents:**\n{sources}"
            else:
                result_to_show = result

            # Display assistant's response
            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
