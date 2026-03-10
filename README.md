## AI Study Copilot

AI Study Copilot is a Retrieval-Augmented Generation (RAG) based study assistant that allows users to upload PDF documents and ask questions about their content.

The system retrieves relevant document chunks and generates answers using a language model. It also provides source citations and supports short conversation memory for follow-up questions.

This project demonstrates a complete RAG pipeline with hybrid retrieval and basic observability, designed as an AI engineering learning project.

## Features

Upload and parse PDF documents

Text cleaning and chunking

Local embedding generation

FAISS vector search

TF-IDF keyword search

Hybrid retrieval (vector + keyword)

Document-based question answering

Source citation with page numbers

Conversation memory

Index caching for faster queries

Basic observability logs (retrieval time and LLM time)

## System Architecture

The system follows a typical Retrieval-Augmented Generation workflow:

PDF Upload
↓
Text Extraction
↓
Text Chunking
↓
Embedding Generation
↓
Vector Index (FAISS)
+
Keyword Index (TF-IDF)
↓
Hybrid Retrieval
↓
LLM Answer Generation
↓
Source Citation

## Project Structure
ai-study-copilot
│
├── app.py
├── text_chunker.py
├── embedding_store.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md

app.py
Main Streamlit application.

text_chunker.py
Handles text cleaning and chunk splitting.

embedding_store.py
Builds the vector index, keyword index, and hybrid retrieval logic.

## Installation

Clone the repository:
git clone https://github.com/yourname/ai-study-copilot.git
cd ai-study-copilot
Install dependencies:
pip install -r requirements.txt

## Run the Project
Start the Streamlit app:
streamlit run app.py
Then open your browser and visit:
http://localhost:8501
Upload a PDF and start asking questions about the document.

## Example Workflow

1.Upload a PDF document

2.The system extracts and cleans text

3.The text is split into chunks

4.Embeddings are generated for each chunk

5.FAISS builds a vector index

6.TF-IDF builds a keyword index

7.Hybrid retrieval selects relevant chunks

8.The LLM generates answers based on retrieved context

9.The system shows answers with source citations

## Tech Stack

Python
Streamlit
FAISS
Sentence Transformers
Scikit-learn (TF-IDF)
DeepSeek API

## Observability

The system logs basic runtime metrics for each query:

Retrieval time

LLM generation time

Total response time

Retrieved chunk information

These logs help analyze system behavior and debug retrieval performance.

## Future Improvements

Possible extensions for this project include:

Multi-document knowledge base

Persistent vector database

Improved retrieval ranking

Web deployment

UI improvements

Agent-based workflows

## License

This project is intended for educational and experimental purposes.