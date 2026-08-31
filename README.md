# 🏠 Real Estate LangChain RAG Assistant

An AI-powered Real Estate Question Answering system built using
Retrieval-Augmented Generation (RAG) and LangChain.

# 📌 Project Overview

This application allows users to ask questions about real estate
documents and receive accurate answers based on the information
available in those documents.

The system uses LangChain for the RAG pipeline, document chunking,
Hugging Face embeddings, FAISS vector search, and Google Gemini
for answer generation.

# 🚀 Features

- 📄 PDF document processing
- ✂️ Text chunking
- 🔢 Hugging Face text embeddings
- 🔎 Semantic search using FAISS
- 🔗 LangChain RAG pipeline
- 🤖 Google Gemini for answer generation
- 📊 RAG evaluation metrics
- 🌐 Streamlit web interface

# 🛠️ Technologies Used

- Python
- LangChain
- Streamlit
- Google Gemini API
- Hugging Face Sentence Transformers
- FAISS
- NumPy
- PyPDF
- python-dotenv

# 🔄 RAG Workflow

PDF Documents
↓
Document Loading
↓
Text Chunking
↓
Hugging Face Embeddings
↓
FAISS Vector Store
↓
User Question
↓
Retriever
↓
Relevant Documents
↓
Context + Question
↓
Google Gemini
↓
Final Answer
↓
RAG Evaluation


# 📊 RAG Evaluation

The generated answers are evaluated using:

- Faithfulness
- Relevance
- Context Relevance

# 📂 Project Structure

Real-Estate-LangChain-RAG/
│
├── .gitignore
├── requirements.txt
├── README.md
│
├── data/
│   └── RealEstate/
│       ├── 01_SkyMeadows_Price_List.pdf
│       ├── 02_EmeraldEnclave_Handbook.pdf
│       ├── 03_Listing_Portfolio.pdf
│       └── 04_Maintenance_Compliance_Log.pdf
│
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
│
└── src/
    ├── config.py
    ├── loader.py
    ├── splitter.py
    ├── embeddings.py
    ├── vector_store.py
    ├── query.py
    ├── generator.py
    ├── evaluation.py
    ├── rag_chain.py
    └── app.py
