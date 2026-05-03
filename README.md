# Multi-Document RAG with Groq

A Retrieval-Augmented Generation (RAG) application that processes multiple documents and answers questions using Groq's LLM API.

## Project Structure

```
multi-doc-rag-groq/
├── src/
│   ├── app.py            # Streamlit web application
│   └── rag_pipeline.py   # Multi-file RAG pipeline logic
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

## Features

- Upload and process multiple documents simultaneously
- Semantic search across all uploaded files
- Fast LLM inference powered by Groq
- Interactive Streamlit UI

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your Groq API key:
```
GROQ_API_KEY=your_api_key_here
```

## Run

```bash
streamlit run src/app.py
```

## Tech Stack

- Python
- Streamlit
- Groq API
- LangChain / RAG pipeline
