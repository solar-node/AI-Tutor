# 📚 AI Tutor - Your Personal Document Expert

**Live App:** [AI Tutor on Streamlit](https://ai-tutor-solarnode.streamlit.app/)  
**Repository:** [solar-node/AI-Tutor](https://github.com/solar-node/AI-Tutor)

---

## 🧠 Overview

**AI Tutor** is a sophisticated application that can teach you concepts from **any PDF document** you upload.  
Built using **Python**, **Streamlit**, and the powerful **LangChain** framework, it uses **Google’s Gemini model** as its reasoning engine.

This project demonstrates the power of combining **document understanding**, **retrieval-augmented generation (RAG)**, and **intelligent agent-based reasoning** to create an interactive personal tutor experience.

---

## ✨ Key Features

- **Dynamic Document Upload** – Upload any PDF textbook, research paper, or document to serve as the knowledge base.  
- **Conversational Chat Interface** – Interact naturally with the AI tutor; it remembers your chat context for continuity.  
- **Retrieval-Augmented Generation (RAG)** – The AI’s primary source of truth is your document, not pre-trained data.  
- **Web Search Fallback** – If your query isn’t in the uploaded material or involves current events, the tutor searches the web using **SerpAPI**.  
- **Intelligent Agent Architecture** – A LangChain “ReAct” (Reasoning and Acting) agent decides whether to use your document or the web for the best answer.

---

## 🛠️ How It Works (Architecture)

This application follows a **modern agent-based RAG architecture**, combining flexible reasoning and efficient retrieval.

### **Frontend**
- Built with **Streamlit** for a fast, responsive chat interface.

### **Document Processing**
When you upload a PDF:
1. The document is loaded using `PyMuPDFLoader`.  
2. Text is split into manageable chunks via `RecursiveCharacterTextSplitter`.  
3. Each chunk is converted into **embeddings** using a Hugging Face model (`BAAI/bge-small-en-v1.5`).  
4. The embeddings are stored in an in-memory **FAISS vector store** for fast, session-based retrieval.

### **The Agent’s Brain**
- The reasoning engine is a **LangChain Agent** powered by **Google’s Gemini-2.5-Flash** model.

### **The Toolbox**
- **Custom Document Tool** – Retrieves relevant chunks from the FAISS vector store.  
- **Web Search Tool** – Uses `SerpAPIWrapper` for real-time information retrieval from the web.

---

## 🚀 Getting Started Locally

Follow these steps to run the project on your own machine.

### 1. Clone the Repository
```bash
git clone https://github.com/solar-node/AI-Tutor.git
cd AI-Tutor
```

### 2. Set Up Your Environment
Create a `.env` file in the root directory and add your API keys:
```bash
GOOGLE_API_KEY=your_google_gemini_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
SERPAPI_API_KEY=your_serpapi_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
```

It’s recommended to use a virtual environment:
```bash
python -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```bash
streamlit run app.py
```

Once it launches, your browser should open the AI Tutor interface automatically.

---

## 💬 Usage

1. Launch the app and wait for it to initialize.  
2. Upload a PDF using the sidebar uploader.  
3. Wait for processing (you’ll see progress messages).  
4. Once it’s ready, type your questions in the chat input at the bottom and press **Enter**.  
5. The tutor will answer contextually, referencing the uploaded document or web results when needed.

---

## 📁 File Structure

```
AI-Tutor/
├── app.py              # Main Streamlit application (UI + logic)
├── requirements.txt    # Project dependencies
├── .env                # (To be created by user) Stores secret API keys
└── README.md           # Project documentation
└── Notebooks           # Jupyter notebooks used for testing
```

---

## 🧩 Technologies Used

- **Python 3.10+**  
- **Streamlit** – UI Framework  
- **LangChain** – Framework for LLM orchestration  
- **Google Gemini (via langchain-google-genai)** – LLM engine  
- **Hugging Face Embeddings (BAAI/bge-small-en-v1.5)** – Vector representation  
- **FAISS** – Vector store for retrieval  
- **SerpAPI** – Web search integration  
- **PyMuPDF** – PDF text extraction  

---

## 💡 Why This Project Matters

This project serves as a practical, open-source example of building **document-aware AI systems** capable of reasoning over uploaded content.  
It’s ideal for students, researchers, or developers interested in:
- Building **RAG systems**  
- Experimenting with **LLM tool usage (agents)**  
- Exploring **AI in education and tutoring**

---

> 🧩 *AI Tutor shows how far personal AI assistants have come—from simple chatbots to full-fledged learning companions.*
