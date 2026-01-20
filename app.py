import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pymupdf

import streamlit as st
from dotenv import load_dotenv

# Setting up the envirinment
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage

from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.pydantic_v1 import BaseModel as CoreBaseModel, Field
from langchain.tools import tool 


# Initializing the models
# @st.cache_resource - To prevent reinitializing on every interaction
@st.cache_resource 
def load_llm():
    return ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash",
        temperature = 0.5
    )

# Embedding model
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEndpointEmbeddings(
        model = "BAAI/bge-small-en-v1.5"
    )

llm = load_llm()
embedding_model = load_embedding_model()

## Backend Logic

# Document processing

# Streamlit will only rerun it if uploaded file's details change
@st.cache_resource
def process_uploaded_book(uploaded_file):
    # Read the uploaded file's bytes and save it temporarily
    bytes_data = uploaded_file.getvalue()
    file_path = os.path.join("./", uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(bytes_data)

    st.write("Extracting text from PDF using PyMuPDF...")
    full_text_content = ""
    with pymupdf.open(file_path) as doc:
        for page in doc:
            full_text_content += page.get_text()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(full_text_content)
    
    # Create the in-memory vector store using FAISS.from_texts
    st.write("Creating in-memory vector store...")
    # Use from_texts because we now have a list of simple text strings
    vector_store = FAISS.from_texts(texts=chunks, embedding=embedding_model)

    os.remove(file_path) # Clean up temperory file
    return vector_store



@st.cache_resource
def create_tutor_agent(_vector_store):
    """"Creates the AI tutor agent with its tools"""
    simple_retriever = _vector_store.as_retriever(search_kwargs = {"k" : 5})

    def textbook_search(query: str) -> str:
        """
        Search the uploaded textbook vector store
        and return combined text of relevant chunks.
        """
        docs = simple_retriever.get_relevant_documents(query)
        # Join the retrieved content into one string
        return "\n\n".join([doc.page_content for doc in docs])  
  
    textbook_search_tool = Tool(
        name="textbook_search",
        func=textbook_search,
        description=(
            "Searches the uploaded textbook for relevant content "
            "about the user's question, definitions, and concepts."
        )
    )

    search = SerpAPIWrapper()
    web_search_tool = Tool(
        name = "web_search",
        func = search.run,
        description = "A useful tool for searching the internet to answer questions about current events, real-world examples, or topics not found in the user's uploaded document."
    )

    tools = [textbook_search_tool, web_search_tool]

    # Base prompt
    base_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI tutor that reasons step by step."),
        ("human", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])

    # System prompt
    system_message = """
        You are an expert AI Tutor. Your personality is helpful, encouraging, and patient. Your primary goal is to help students understand concepts from their uploaded document.
        
        **Your Operational Rules:**
        
        1.  **Prioritize the Textbook:** When a student asks a question, your first action **MUST** be to use the `textbook_search` tool. This is your primary source of truth.
        2.  **Web Search as a Fallback:** You should only use the `web_search` tool in two specific situations:
            * If the `textbook_search` tool returns no relevant information or you determine the answer is insufficient.
            * If the student explicitly asks for real-world examples, current events, or information clearly outside the scope of the textbook.
        3.  **Synthesize, Don't Just Report:** Never just copy-paste the output from a tool. Always synthesize the information into a clear, easy-to-understand explanation. Use analogies and simple terms.
        4.  **Engage the Student:** After explaining a concept, always end your response with a follow-up question to check for understanding or encourage further discussion. For example: "Does that make sense?" or "What part of that would you like to explore further?".
        5.  **Acknowledge Your Source:** If you use the web search tool, briefly mention it. For example: "That's a great question that goes beyond the textbook. According to a quick search...
    """

    agent = create_react_agent(
        llm,
        tools,
        base_prompt
    )

    agent_executor = AgentExecutor(
        agent = agent,
        tools = tools,
        verbose = True,
        handle_parsing_errors = True
    )
    
    return agent_executor



# Streamlit UI 
st.title("📚 AI Tutor from Your Textbook")
st.write("Upload a pdf document in the sidebar and start asking questions.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None

with st.sidebar:
    st.header("Your Document")
    uploaded_file = st.file_uploader("Upload your Textbook", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        if st.session_state.agent_executor is None:
            with st.spinner("Processing your document...This may take a moment."):
                vector_store = process_uploaded_book(uploaded_file)
                st.session_state.agent_executor = create_tutor_agent(vector_store)
            
            st.success("Document processed! You can now ask questions...")


if st.session_state.agent_executor:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

prompt = st.chat_input("Ask a question about your book")

if prompt:
    st.session_state.messages.append({
        "role" : "user",
        "content" : prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # history = [
            #     HumanMessage(
            #         content = msg["content"]) if msg["role"] == "user"
            #         else  AIMessage(content = msg["content"])
            #         for message in st.session_state.messages[:-1]
            # ]

            history = []

            # Go through all previous messages except the latest one
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    history.append(HumanMessage(content=msg["content"]))
                else:
                    history.append(AIMessage(content=msg["content"]))
            
            response = st.session_state.agent_executor.invoke({
                "input" : prompt,
                "chat_history" : history
            })

            tutor_response = response['output']
            st.markdown(tutor_response)
    
    st.session_state.messages.append({"role" : "assistant", "content" : tutor_response})

else:
    st.info("Please upload a pdf document in the sidebar to begin.")




