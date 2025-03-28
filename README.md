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

## Prerequisites

- Python 3.9+
- OpenAI API key and/or Anthropic API key
- For PDF processing:
  - Windows: No additional requirements
  - Linux: `tesseract-ocr` and `poppler-utils`
  ```bash
  # Ubuntu/Debian
  sudo apt-get update && sudo apt-get install -y tesseract-ocr poppler-utils
  ```

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd rag-llm-chat
```

2. Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

## Running Locally

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Deployment

### Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy from your repository
5. Add secrets in the Streamlit Cloud dashboard:
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`

### Docker Deployment

1. Build the image:

```bash
docker build -t rag-llm-chat .
```

2. Run the container:

```bash
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your_key \
  -e ANTHROPIC_API_KEY=your_key \
  rag-llm-chat
```

## Usage

1. Open the app in your browser
2. Enter your API keys (if not set in environment)
3. Select your preferred model
4. Toggle RAG mode if you want to use document-based responses
5. Upload documents or paste URLs for RAG context
6. Start chatting!

## Project Structure

```
rag-llm-chat/
├── app.py              # Main Streamlit application
├── rag_methods.py      # RAG functionality implementation
├── requirements.txt    # Project dependencies
├── .env               # Environment variables (not in repo)
└── README.md          # This file
```

## Dependencies

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
