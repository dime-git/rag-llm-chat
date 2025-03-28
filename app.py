import streamlit as st
import os
import dotenv
import uuid

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage

dotenv.load_dotenv()

MODELS = [
    "openai/o1-mini",
    "openai/gpt-4o"
    "claude-3-5-sonnet-20240620"
]

set.set_page_config(
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
