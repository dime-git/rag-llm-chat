import streamlit as st
import os
import dotenv
from time import time
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
from chromadb.config import Settings
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

dotenv.load_dotenv()

DB_DOCS_LIMIT = 10

#function to stream the response of the LLM
def stream_llm_response(llm_stream, messages):
    for chunk in llm_stream.stream(messages):
        if hasattr(chunk, 'content'):
            yield chunk.content


##indexing phase 
def load_doc_to_db():
    if "rag_docs" not in st.session_state or not st.session_state.rag_docs:
        return

    if "rag_sources" not in st.session_state:
        st.session_state.rag_sources = []

    docs = []
    processed_files = []
    for doc_file in st.session_state.rag_docs:
        try:
            print(f"Processing file: {doc_file.name}, Type: {doc_file.type}")  # Debug log
            
            if doc_file.name in st.session_state.rag_sources:
                print(f"File {doc_file.name} already processed, skipping...")
                continue

            if len(st.session_state.rag_sources) >= DB_DOCS_LIMIT:
                st.error(f"Maximum number of documents reached ({DB_DOCS_LIMIT})")
                break

            # Create source_files directory if it doesn't exist
            os.makedirs("source_files", exist_ok=True)
            file_path = os.path.join("source_files", doc_file.name)
            
            # Save uploaded file
            try:
                content = doc_file.getvalue()
                print(f"File size: {len(content)} bytes")  # Debug log
                with open(file_path, "wb") as file:
                    file.write(content)
            except Exception as e:
                st.error(f"Error saving file {doc_file.name}: {str(e)}")
                print(f"Error saving file: {str(e)}")
                continue

            # Load document based on type
            try:
                if doc_file.type == "application/pdf":
                    loader = PyPDFLoader(file_path)
                elif doc_file.name.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                elif doc_file.type in ["text/plain", "text/markdown"]:
                    loader = TextLoader(file_path)
                else:
                    st.warning(f"Document type {doc_file.type} not supported")
                    continue

                # Load and add to docs list
                loaded_docs = loader.load()
                print(f"Successfully loaded {len(loaded_docs)} documents from {doc_file.name}")  # Debug log
                docs.extend(loaded_docs)
                st.session_state.rag_sources.append(doc_file.name)
                processed_files.append(doc_file.name)

            except Exception as e:
                st.error(f"Error loading document {doc_file.name}: {str(e)}")
                print(f"Error loading document: {str(e)}")
                continue

            finally:
                # Clean up temporary file
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error removing temporary file: {str(e)}")

        except Exception as e:
            st.error(f"Unexpected error processing {doc_file.name}: {str(e)}")
            print(f"Unexpected error: {str(e)}")
            continue

    # Process loaded documents
    if docs:
        try:
            print(f"Processing {len(docs)} documents...")  # Debug log
            _split_and_load_docs(docs)
            if processed_files:
                st.success(f"Documents loaded successfully: {', '.join(processed_files)}")
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            print(f"Error processing documents: {str(e)}")
    else:
        st.warning("No documents were successfully loaded.")


def load_url_to_db():
    if not st.session_state.get("rag_url"):
        return

    url = st.session_state.rag_url.strip()
    if not url:
        return

    if "rag_sources" not in st.session_state:
        st.session_state.rag_sources = []

    print(f"Processing URL: {url}")  # Debug log

    if url in st.session_state.rag_sources:
        st.warning(f"URL {url} already processed")
        return

    if len(st.session_state.rag_sources) >= DB_DOCS_LIMIT:
        st.error(f"Maximum number of documents reached ({DB_DOCS_LIMIT})")
        return

    docs = []
    try:
        loader = WebBaseLoader(url)
        loaded_docs = loader.load()
        print(f"Successfully loaded content from URL: {url}")  # Debug log
        docs.extend(loaded_docs)
        st.session_state.rag_sources.append(url)

        if docs:
            _split_and_load_docs(docs)
            st.success(f"Successfully loaded content from: {url}")
            # Clear the URL input after successful loading
            st.session_state.rag_url = ""
        else:
            st.warning("No content could be extracted from the URL")

    except Exception as e:
        st.error(f"Error loading content from {url}: {str(e)}")
        print(f"Error loading URL: {str(e)}")  # Debug log


def _split_and_load_docs(docs):
    if not docs:
        st.warning("No documents to process.")
        return

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 5000,
            chunk_overlap = 1000
        )

        print(f"Splitting {len(docs)} documents into chunks...")  # Debug log
        documents_chunks = text_splitter.split_documents(docs)
        print(f"Created {len(documents_chunks)} chunks")  # Debug log

        # Always create a new vector store instance for the chunks
        vector_db = initialize_vector_db(documents_chunks)
        if vector_db is not None:
            st.session_state.vector_db = vector_db
            print("Vector database initialized successfully")
        else:
            st.error("Failed to initialize vector database. Please check your API key and try again.")
            return

    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        print(f"Error in _split_and_load_docs: {str(e)}")
        return

def initialize_vector_db(documents_chunks):
    try:
        # Check if OpenAI API key is available
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            st.error("OpenAI API key not found. Please add your API key in the sidebar.")
            return None

        print("Initializing vector database...")  # Debug log
        
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Create an in-memory Chroma instance
        vector_db = Chroma(
            collection_name=f"collection_{str(time()).replace('.', '')[:14]}",
            embedding_function=embeddings,
        )
        
        # Add documents to the collection
        vector_db.add_documents(documents_chunks)
        
        return vector_db
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg:
            st.error("Invalid OpenAI API key. Please check your API key in the sidebar.")
        else:
            st.error(f"Error initializing vector database: {error_msg}")
        print(f"Error initializing vector database: {error_msg}")
        return None


# Retrieve

def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get inforamtion relevant to the conversation, focusing on the most recent messages."),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

def get_conversational_rag_chain(llm):
    """Create a conversational RAG chain"""
    try:
        retriever = st.session_state.vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  # Specify the output key
        )

        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=True,
            return_generated_question=False,
            output_key="answer"  # Specify the output key
        )
    except Exception as e:
        error_msg = f"Error creating RAG chain: {str(e)}"
        print(error_msg)  # Debug log
        raise Exception(error_msg)

def stream_llm_rag_response(llm, messages):
    """Stream responses from the LLM using RAG"""
    try:
        if not st.session_state.get('vector_db'):
            raise ValueError("Vector database not initialized. Please upload documents first.")

        print("Creating RAG chain...")  # Debug log
        qa_chain = get_conversational_rag_chain(llm)
        
        # Get the last user message
        last_user_message = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        if not last_user_message:
            raise ValueError("No user message found in conversation")

        print("Streaming RAG response...")  # Debug log
        response = qa_chain({"question": last_user_message})
        
        # Extract just the answer from the response
        if isinstance(response, dict) and "answer" in response:
            yield response["answer"]
        else:
            raise ValueError("Unexpected response format from RAG chain")
                
    except Exception as e:
        error_msg = f"Error in RAG response: {str(e)}"
        print(error_msg)  # Debug log
        st.error(error_msg)
        yield f"I encountered an error: {str(e)}"