import streamlit as st
import os
import dotenv
import uuid

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage

from rag_methods import (
    load_url_to_db,
    load_doc_to_db,
    stream_llm_response,
    stream_llm_rag_response,
)

dotenv.load_dotenv()

MODELS = [
    # "openai/o1-mini",
    "openai/gpt-4o",
    "claude-3-5-sonnet-20240620"
]

st.set_page_config(
    page_title = "RAG LLM app",
    layout = "centered",
    initial_sidebar_state = "expanded"
)

st.html("""<h2 style="text-align: center;"><i>Do you LLM even RAG broski?<i></h2>""")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello there! How can I assist you today?"
    }]

with st.sidebar:
    default_openai_api_key = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") is not None else "" #only for dev env
    with st.popover("OpenAI"):
        open_ai_api_key = st.text_input(
            "Introduce your OpenAI API Key (http://platform.openai.com/)",
            value=default_openai_api_key,
            type="password"
        )

with st.sidebar:
    default_anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") if os.getenv("ANTHROPIC_API_KEY") is not None else "" #only for dev env
    with st.popover("Anthropic"):
        anthropic_ai_api_key = st.text_input(
            "Introduce your Anthropic API Key (http://console.anthropic.com/)",
            value=default_anthropic_api_key,
            type="password"
        )

#Checking if the user has introduced the OpenAI API Key, if not, a warning is displayed
missing_openai_key = open_ai_api_key == "" or open_ai_api_key is None or "sk-" not in open_ai_api_key
missing_anthropic_key = anthropic_ai_api_key == "" or anthropic_ai_api_key is None
if missing_openai_key and missing_anthropic_key:
    st.write("#")
    st.warning("Please enter your API Key to continue...")

else:
    #sidebar
    with st.sidebar:
        st.divider()
        st.selectbox(
            "Select a model",
            [model for model in MODELS if ("openai" in model and not missing_openai_key) or ("anthropic" in model and not missing_anthropic_key)], key="model"
        )
        cols = st.columns(2)
        with cols[0]:
            is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)
            st.toggle(
                "Use RAG",
                value=is_vector_db_loaded,
                key="use_rag",
                disabled=not is_vector_db_loaded
            ) 

        with cols[1]:
            st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

        st.header("RAG Sources:")

        #File upload input for RAG with documents
        st.file_uploader(
            "Upload document",
            type=["pdf","txt","docx","md"],
            accept_multiple_files=True,
            on_change=load_doc_to_db,
            key="rag_docs"
        )

        st.text_input(
            "Paste your URL",
            placeholder="https://example.com",
            on_change=load_doc_to_db,
            key="rag_url"
        )

        with st.expander(f"Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.vector_db.get()['metadatas'])})"):
            st.write([] if not is_vector_db_loaded else [meta['source'] for meta in st.session_state.vector_db.get()['metadatas']])

    #main chat app
    model_provider = st.session_state.model.split("/")[0]
    if model_provider == "openai":
        llm_stream = ChatOpenAI(
            model_name = st.session_state.model.split("/")[-1],
            temperature=0.3,
            streaming=True
        )
    elif model_provider == "anthropic":
        llm_stream = ChatAnthropic(
            model_name = st.session_state.model.state("/")[-1],
            temperature=0.3,
            streaming=True
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]
            message_placeholder = st.empty()
            response = ""
            
            if not st.session_state.use_rag:
                for chunk in stream_llm_response(llm_stream, messages):
                    response += chunk
                    message_placeholder.markdown(response)
            else:
                for chunk in stream_llm_rag_response(llm_stream, messages):
                    response += chunk
                    message_placeholder.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
