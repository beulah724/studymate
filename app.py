import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai import Credentials
from dotenv import load_dotenv
import os

# ----------------- LOAD ENVIRONMENT VARIABLES -----------------
load_dotenv()
api_key = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("WATSONX_PROJECT_ID")

if not api_key or not project_id:
    st.error("⚠️ Please add WATSONX_API_KEY and WATSONX_PROJECT_ID in your .env file.")
    st.stop()

creds = Credentials(
    api_key=api_key,
    url="https://us-south.ml.cloud.ibm.com"
)

# ----------------- IBM WATSONX MODEL -----------------
llm_model = Model(
    model_id="mistralai/mixtral-8x7b-instruct",
    credentials=creds,
    params={"decoding_method": "greedy", "max_new_tokens": 500},
    project_id=project_id
)

# ----------------- EMBEDDING MODEL -----------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------- FUNCTIONS -----------------
def read_pdf(file):
    """Extract text from PDF file"""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text_data = ""
    for page in doc:
        text_data += page.get_text("text")
    return text_data

def split_text(text, chunk_size=500):
    """Split text into smaller chunks"""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def create_index(chunks):
    """Create FAISS index from text chunks"""
    vectors = embedder.encode(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))
    return index, vectors

def search_chunks(query, chunks, index):
    """Retrieve best matching chunks"""
    q_vector = embedder.encode([query])
    _, idx = index.search(np.array(q_vector), k=3)
    return [chunks[i] for i in idx[0]]

def generate_answer(question, context):
    """Generate AI answer from context using Watsonx"""
    prompt = f"""
    You are StudyMate, a helpful AI for students.
    Context: {context}
    Question: {question}
    Answer clearly and cite the content when possible.
    """
    response = llm_model.generate_text(prompt=prompt)
    return response

# ----------------- STREAMLIT UI -----------------
st.set_page_config(page_title="StudyMate", page_icon="📘", layout="wide")
st.title("📘 StudyMate - Your AI Study Assistant")

uploaded_files = st.file_uploader("Upload your PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_chunks = []
    for file in uploaded_files:
        text = read_pdf(file)
        chunks = split_text(text)
        all_chunks.extend(chunks)

    st.success("✅ PDFs processed successfully!")
    index, vectors = create_index(all_chunks)

    user_question = st.text_input("Ask a question from your study materials:")
    if st.button("Get Answer") and user_question:
        results = search_chunks(user_question, all_chunks, index)
        context = " ".join(results)
        answer = generate_answer(user_question, context)
        st.markdown(f"### 🤖 Answer:\n{answer}")
        st.markdown("**📚 Sources:**")
        for r in results:
            st.write(f"- {r[:300]}...")
