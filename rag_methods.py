import streamlit as st
import os
import dotenv
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
)

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

dotenv.load_dotenv()

DB_DOCS_LIMIT = 10

#function to stream the response of the LLM
def stream_llm_response(llm_stream, messages):
    full_response = ""

    for chunk in llm_stream.stream(messages):
        if hasattr(chunk, 'content'):
            full_response += chunk.content
            yield chunk.content
    
    return full_response