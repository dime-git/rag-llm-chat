# RAG LLM Chat

A powerful chat application that combines Large Language Models (LLMs) with Retrieval Augmented Generation (RAG) capabilities. Built with Streamlit, supporting both OpenAI and Anthropic models.

## Features

- 🤖 Multi-model support (OpenAI GPT-4 and Anthropic Claude)
- 📚 RAG capabilities with document processing
- 📄 Support for multiple file formats (PDF, TXT, DOCX, MD)
- 🔗 URL content ingestion
- ⚡ Streaming responses
- 🔒 Secure API key management
- 💾 Vector database for efficient document retrieval

## Project Structure

```
rag-llm-chat/
├── app.py              # Main Streamlit application
├── rag_methods.py      # RAG functionality implementation
├── requirements.txt    # Project dependencies
├── .env               # Environment variables (not in repo)
└── README.md          # This file
```

## Dependencies used

Key dependencies include:

- `streamlit`: Web application framework
- `langchain`: LLM framework and tools
- `chromadb`: Vector store for document embeddings
- `anthropic`: Claude model integration
- `openai`: OpenAI model integration
- `unstructured`: Document processing
- `python-magic`: File type detection
- `pdf2image`: PDF processing
- `pytesseract`: OCR capabilities
