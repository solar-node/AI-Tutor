import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pymupdf
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent  # Updated import


@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.5
    )


@st.cache_resource
def load_embedding_model():
    return HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-small-en-v1.5"
    )


llm = load_llm()
embedding_model = load_embedding_model()


@st.cache_resource
def process_uploaded_book(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    file_path = os.path.join("./", uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(bytes_data)

    st.write("Extracting text from PDF using PyMuPDF...")
    full_text_content = ""
    with pymupdf.open(file_path) as doc:
        for page in doc:
            full_text_content += page.get_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(full_text_content)

    st.write("Creating in-memory vector store...")
    vector_store = FAISS.from_texts(texts=chunks, embedding=embedding_model)

    os.remove(file_path)
    return vector_store


@st.cache_resource
def create_tutor_agent(_vector_store):
    """Creates the AI tutor agent with its tools"""
    simple_retriever = _vector_store.as_retriever(search_kwargs={"k": 5})

    @tool("textbook_search")
    def textbook_search(query: str) -> str:
        """
        Search the uploaded textbook vector store
        and return combined text of relevant chunks.
        Use this tool FIRST for any question about the textbook content.
        """
        docs = simple_retriever.invoke(query)
        if not docs:
            return "NO_RELEVANT_TEXT_FOUND"
        return "\n\n".join(doc.page_content for doc in docs)

    @tool("web_search")
    def web_search(query: str) -> str:
        """
        Search the internet for current events, real-world examples,
        or topics not found in the uploaded document.
        Only use this if textbook_search returns insufficient results.
        """
        search = SerpAPIWrapper()
        return search.run(query)

    tools = [textbook_search, web_search]

    system_message = """You are an expert AI Tutor. Your personality is helpful, encouraging, and patient. 
Your primary goal is to help students understand concepts from their uploaded document.

**Your Operational Rules:**
1. **Prioritize the Textbook:** When a student asks a question, your first action MUST be to use the `textbook_search` tool.
2. **Web Search as a Fallback:** Only use `web_search` if the textbook is insufficient or for real-world examples.
3. **Synthesize, Don't Just Report:** Explain concepts simply.
4. **Engage the Student:** End with a follow-up question like "Does that make sense?".
5. **Acknowledge Your Source:** Mention if you used web search."""

    # Use the new create_agent from langchain.agents
    agent_executor = create_agent(
        model=llm,
        tools=tools,
        prompt=system_message,  # Use 'prompt' parameter
    )

    return agent_executor


# Streamlit UI
st.title("📚 AI Tutor from Your Textbook")
st.write("Upload a PDF document in the sidebar and start asking questions.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None

with st.sidebar:
    st.header("Your Document")
    uploaded_file = st.file_uploader(
        "Upload your Textbook", type="pdf", label_visibility="collapsed"
    )

    if uploaded_file:
        if st.session_state.agent_executor is None:
            with st.spinner("Processing your document... This may take a moment."):
                vector_store = process_uploaded_book(uploaded_file)
                st.session_state.agent_executor = create_tutor_agent(vector_store)

            st.success("Document processed! You can now ask questions...")

# Display chat history
if st.session_state.agent_executor:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about your book")

    if prompt:
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Build message history
                history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        history.append(HumanMessage(content=msg["content"]))
                    else:
                        history.append(AIMessage(content=msg["content"]))

                # Add current message
                history.append(HumanMessage(content=prompt))

                # Invoke with messages
                response = st.session_state.agent_executor.invoke({
                    "messages": history
                })

                # Extract response
                if "messages" in response:
                    last_message = response["messages"][-1]
                    tutor_response = last_message.content
                elif "output" in response:
                    tutor_response = response["output"]
                else:
                    tutor_response = str(response)

                st.markdown(tutor_response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": tutor_response
        })

else:
    st.info("Please upload a PDF document in the sidebar to begin.")