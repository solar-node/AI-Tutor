import os
import streamlit as st
from dotenv import load_dotenv

# Setting up the envirinment
load_dotenv()

from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_docling import DoclingLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from docling.document_converter import DocumentConverter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage



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

    # Converting pdf to markdown using docling
    st.write("converting PDF to structured markdown using docling")
    converter = DocumentConverter()
    doc = converter.convert(file_path).document
    full_markdown_content = doc.export_to_markdown()

    # Split the markdown content based on headers
    headers_to_split_on = [
        ("## PART", "Part"),
        ("## CHAPTER", "Chapter"),
        ("## ", "Section"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on = headers_to_split_on,
        strip_headers = False
    )
    chunks = markdown_splitter.split_text(full_markdown_content)

    # Create the in-memory vector store
    st.write("Creating in memory vector store")
    vector_store = FAISS.from_documents(chunks, embedding_model)

    os.remove(file_path) # Clean up temperory file
    return vector_store



@st.cache_resource
def create_tutor_agent(vector_store):
    """"Creates the AI tutor agent with its tools"""
    simple_retriever = vector_store.as_retriever(search_kwargs = {"k" : 5})

    textbook_search_tool = create_retriever_tool(
        simple_retriever,
        name = "textbook_search",
        description = "This is the primary tool for answering questions. Use it to search the user's uploaed textbook for specific topics, definitions, and concepts. Always use this tool."
    )

    search = SerpAPIWrapper()
    web_search_tool = Tool(
        name = "web_search",
        func = search.run,
        description = "A useful tool for searching the internet to answer questions about current events, real-world examples, or topics not found in the user's uploaded document."
    )

    tools = [textbook_search_tool, web_search_tool]

    # Base prompt
    base_prompt = hub.pull("hwchase17/react-chat")

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




