# connect_memory_with_llm.py

import os
from dotenv import load_dotenv, find_dotenv

from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_environment():
    """
    Load environment variables from .env file.
    """
    load_dotenv(find_dotenv())
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment variables.")
    return hf_token


def load_llm(huggingface_repo_id: str, hf_token: str) -> HuggingFaceEndpoint:
    """
    Initialize the HuggingFace LLM endpoint.
    """
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={
            "token": hf_token,
            "max_length": 512  # Integer value
        }
    )
    return llm


def set_custom_prompt(template: str) -> PromptTemplate:
    """
    Define a custom prompt template.
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    return prompt


def main():
    # Load environment variables
    hf_token = load_environment()

    HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

    # Define custom prompt
    CUSTOM_PROMPT_TEMPLATE = """
    Use the pieces of information provided in the context to answer the user's question.
    If you don't know the answer, just say that you don't know; don't try to make up an answer. 
    Don't provide anything out of the given context.

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """

    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load FAISS vector store
    DB_FAISS_PATH = "vectorstore/db_faiss"
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

    # Initialize LLM
    llm = load_llm(HUGGINGFACE_REPO_ID, hf_token)

    # Initialize RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )

    # Prompt user for input
    user_query = input("Write Query Here: ")
    response = qa_chain.invoke({'question': user_query})
    print("\nRESULT:")
    print(response["result"])
    print("\nSOURCE DOCUMENTS:")
    for doc in response["source_documents"]:
        source = doc.metadata.get('source', 'Unknown Source')
        print(f"- {source}")


if __name__ == "__main__":
    main()
