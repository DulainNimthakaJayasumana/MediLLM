Process PDFs and Create Vector Store:

powershell
Copy
python create_memory_for_llm.py
Purpose: This script processes your PDF documents, creates embeddings, and stores them in a FAISS vector store.

Interact with the Chatbot via Command Line (Optional):

powershell
Copy
python connect_memory_with_llm.py
Purpose: This script allows you to interact with the chatbot through the command line.

Launch the Streamlit Web Application:

powershell
Copy
streamlit run medibot.py
Purpose: This script launches a web-based interface for the chatbot using Streamlit.