import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load the GROQ_API_KEY from the .env file (used when running locally)
load_dotenv()

# ---------- Page setup ----------
st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
st.title("📄 Chat with your PDF")
st.caption("Upload a PDF, then ask questions about it. Answers come from your document.")


# ---------- Build the vector store from an uploaded PDF ----------
# @st.cache_resource means this heavy work runs only once per uploaded file,
# not on every question. This is what keeps the app fast.
@st.cache_resource(show_spinner="Reading and indexing your PDF...")
def build_retriever(file_bytes, file_name):
    # 1. Save the uploaded file to a temporary path so the loader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # 2. LOAD: read the PDF into text
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # 3. CHUNK: split into overlapping passages
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    # 4. EMBED: turn each chunk into a vector (runs locally, no API key needed)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 5. STORE: put the vectors in a FAISS index (fast in-memory similarity search)
    vector_store = FAISS.from_documents(chunks, embeddings)

    os.remove(tmp_path)  # clean up the temp file

    # Return a retriever that fetches the top 4 most relevant chunks
    return vector_store.as_retriever(search_kwargs={"k": 4})


# ---------- The LLM ----------
@st.cache_resource
def get_llm():
    key = None
    try:
        key = st.secrets["GROQ_API_KEY"]
    except Exception:
        key = os.environ.get("GROQ_API_KEY")
    if not key:
        st.error("No GROQ_API_KEY found in secrets or environment.")
        st.stop()
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=key,
    )


# ---------- The prompt template ----------
# We tell the model to answer ONLY from the provided context.
PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
)


# ---------- The answer function (replaces RetrievalQA) ----------
# This does manually what the old chain did: retrieve chunks, stuff them into
# the prompt, and call the LLM. Easy to read and explain.
def answer_question(retriever, llm, question):
    # 1. RETRIEVE the most relevant chunks for this question
    docs = retriever.invoke(question)

    # 2. Join the chunk texts into one context string
    context = "\n\n".join(doc.page_content for doc in docs)

    # 3. Fill the prompt and call the LLM
    messages = PROMPT.format_messages(context=context, question=question)
    response = llm.invoke(messages)

    # Return both the answer text and the source chunks
    return response.content, docs


# ---------- The app flow ----------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Build the retriever (cached, so only runs once for this file)
    retriever = build_retriever(uploaded_file.getvalue(), uploaded_file.name)
    llm = get_llm()

    st.success(f"Ready! Ask anything about '{uploaded_file.name}'.")

    question = st.text_input("Your question:")

    if question:
        with st.spinner("Thinking..."):
            answer, source_docs = answer_question(retriever, llm, question)

        # Show the answer
        st.markdown("### Answer")
        st.write(answer)

        # Show which parts of the PDF the answer came from
        with st.expander("Sources (passages used to answer)"):
            for i, doc in enumerate(source_docs, start=1):
                page = doc.metadata.get("page", "?")
                st.markdown(f"**Source {i} (page {page}):**")
                st.write(doc.page_content[:300] + "...")
else:
    st.info("Upload a PDF above to get started.")