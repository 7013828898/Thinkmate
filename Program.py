import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from collections import defaultdict
import re
import time
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="StudyMate - AI-Powered PDF Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;

        border-left: 5px solid #1E88E5;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 5px solid #4CAF50;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 5px solid #FF9800;
    }
    .answer-box {
        background-color: #F5F5F5;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #424242;
    }
    .reference-box {
        background-color: #ECEFF1;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        font-size: 0.9rem;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #F5F5F5;
    }
    .model-selector {
        background-color: #FFF9C4;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 5px solid #FFD600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'pdf_texts' not in st.session_state:
    st.session_state.pdf_texts = {}
if 'index' not in st.session_state:
    st.session_state.index = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'qa_model' not in st.session_state:
    st.session_state.qa_model = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'chunk_doc_map' not in st.session_state:
    st.session_state.chunk_doc_map = {}
if 'questions_answers' not in st.session_state:
    st.session_state.questions_answers = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Hugging Face model options
MODEL_OPTIONS = {
    "Small & Fast (T5-Small)": "t5-small",
    "Balanced (DistilBERT)": "distilbert-base-cased-distilled-squad",
    "High Quality (BERT Large)": "deepset/bert-large-uncased-whole-word-masking-squad2",
    "Advanced (RoBERTa)": "deepset/roberta-base-squad2"
}

# Function to load the QA model
@st.cache_resource(show_spinner=False)
def load_qa_model(model_name):
    if model_name == "t5-small":
        # Use a text2text generation approach for T5
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return {"tokenizer": tokenizer, "model": model, "type": "t5"}
    else:
        # Use question answering pipeline for other models
        return pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name,
            device=0 if torch.cuda.is_available() else -1
        )

# Function to generate answer using Hugging Face models
def query_huggingface(question, context, model_info):
    try:
        if model_info.get("type") == "t5":
            # Format for T5 model
            input_text = f"question: {question} context: {context}"
            inputs = model_info["tokenizer"](input_text, return_tensors="pt", max_length=512, truncation=True)
            outputs = model_info["model"].generate(
                inputs.input_ids, 
                max_length=200, 
                num_beams=4, 
                early_stopping=True
            )
            answer = model_info["tokenizer"].decode(outputs[0], skip_special_tokens=True)
            return answer
        else:
            # Use QA pipeline for other models
            result = model_info({
                'question': question,
                'context': context[:4000]  # Limit context length to avoid token limits
            })
            return result['answer']
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# App header
st.markdown('<h1 class="main-header">📚 StudyMate</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #555;">AI-Powered PDF-Based Q&A System for Students</p>', unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📁 Upload Study Materials")
    
    uploaded_files = st.file_uploader(
        "Upload PDF documents (textbooks, notes, papers)",
        type="pdf",
        accept_multiple_files=True
    )
    
    if uploaded_files:
        with st.expander("Uploaded Documents", expanded=True):
            for i, file in enumerate(uploaded_files):
                st.write(f"{i+1}. {file.name}")

    st.markdown("---")
    st.markdown("### ⚙️ Processing Options")
    
    chunk_size = st.slider(
        "Text Chunk Size (characters)",
        min_value=500,
        max_value=2000,
        value=1000,
        help="Smaller chunks may provide more precise answers but might lack context"
    )
    
    num_chunks = st.slider(
        "Number of chunks to use for answers",
        min_value=3,
        max_value=10,
        value=5,
        help="More chunks provide broader context but may include less relevant information"
    )
    
    st.markdown("---")
    st.markdown("### 🤖 AI Model Selection")
    
    selected_model = st.selectbox(
        "Choose a question-answering model",
        list(MODEL_OPTIONS.keys()),
        index=1,  # Default to DistilBERT
        help="Larger models provide better answers but require more resources"
    )
    
    if st.button("Load Model"):
        with st.spinner(f"Loading {selected_model}..."):
            st.session_state.qa_model = load_qa_model(MODEL_OPTIONS[selected_model])
            st.session_state.model_loaded = True
            st.success(f"{selected_model} loaded successfully!")

with col2:
    st.markdown("### 💬 Ask Questions About Your Documents")
    
    if not uploaded_files:
        st.markdown('<div class="info-box">📝 Please upload PDF documents to get started. StudyMate will extract text and prepare it for questioning.</div>', unsafe_allow_html=True)
    elif not st.session_state.model_loaded:
        st.markdown('<div class="warning-box">🤖 Please select and load a question-answering model from the left panel.</div>', unsafe_allow_html=True)
    else:
        # Process PDFs if they haven't been processed yet
        if st.session_state.index is None:
            with st.spinner("Processing PDFs and building knowledge base..."):
                # Initialize the embedding model
                try:
                    st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                except:
                    st.error("Error loading embedding model. Please check your internet connection.")
                    st.stop()
                
                # Process each PDF
                all_chunks = []
                chunk_doc_map = {}
                
                for file_idx, uploaded_file in enumerate(uploaded_files):
                    # Extract text from PDF
                    pdf_text = ""
                    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                    
                    for page_num in range(len(pdf_document)):
                        page = pdf_document.load_page(page_num)
                        pdf_text += page.get_text()
                    
                    # Store the extracted text
                    st.session_state.pdf_texts[uploaded_file.name] = pdf_text
                    
                    # Chunk the text
                    chunks = [pdf_text[i:i+chunk_size] for i in range(0, len(pdf_text), chunk_size)]
                    
                    # Add metadata to track which document each chunk came from
                    for chunk_idx, chunk in enumerate(chunks):
                        all_chunks.append(chunk)
                        chunk_doc_map[len(all_chunks)-1] = {
                            'doc_name': uploaded_file.name,
                            'chunk_idx': chunk_idx,
                            'page': (chunk_idx * chunk_size) // len(pdf_text) * len(pdf_document) + 1
                        }
                    
                    pdf_document.close()
                
                # Create embeddings and FAISS index
                if all_chunks:
                    embeddings = st.session_state.embedding_model.encode(all_chunks)
                    dimension = embeddings.shape[1]
                    st.session_state.index = faiss.IndexFlatL2(dimension)
                    st.session_state.index.add(np.array(embeddings).astype('float32'))
                    st.session_state.chunks = all_chunks
                    st.session_state.chunk_doc_map = chunk_doc_map
                    
                    st.markdown(f'<div class="success-box">✅ Processed {len(uploaded_files)} document(s) with {len(all_chunks)} text chunks. You can now ask questions!</div>', unsafe_allow_html=True)
        
        # Question input
        question = st.text_input("Enter your question:", placeholder="e.g., What is the main topic of chapter 3?")
        
        if st.button("Get Answer") and question:
            with st.spinner("Searching documents and generating answer..."):
                # Encode the question
                question_embedding = st.session_state.embedding_model.encode([question])
                
                # Search for similar chunks
                D, I = st.session_state.index.search(np.array(question_embedding).astype('float32'), num_chunks)
                
                # Get the most relevant chunks
                relevant_chunks = [st.session_state.chunks[i] for i in I[0]]
                
                # Prepare context for the LLM
                context = "\n\n".join(relevant_chunks)
                
                # Generate answer using Hugging Face model
                answer = query_huggingface(question, context, st.session_state.qa_model)
                
                # Store Q&A for history
                st.session_state.questions_answers.append({
                    'question': question,
                    'answer': answer,
                    'context_chunks': I[0]
                })
            
            # Display the answer
            st.markdown("### 📝 Answer")
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
            
            # Display references
            st.markdown("### 📚 References")
            for idx, chunk_idx in enumerate(I[0]):
                doc_info = st.session_state.chunk_doc_map[chunk_idx]
                st.markdown(f'<div class="reference-box">Reference {idx+1}: From <b>{doc_info["doc_name"]}</b> (approx. page {doc_info["page"]})</div>', unsafe_allow_html=True)
        
        # Display Q&A history
        if st.session_state.questions_answers:
            st.markdown("---")
            st.markdown("### 📖 Question History")
            
            for i, qa in enumerate(reversed(st.session_state.questions_answers)):
                with st.expander(f"Q: {qa['question']}"):
                    st.markdown(f"**A:** {qa['answer']}")

# Sidebar
with st.sidebar:
    st.markdown("## ℹ️ About StudyMate")
    st.markdown("""
    StudyMate is an AI-powered academic assistant that enables students to interact with their study materials in a conversational format.
    
    **How it works:**
    1. Upload your PDF documents
    2. StudyMate processes and indexes the content
    3. Ask questions about your materials
    4. Get AI-generated answers with references
    
    **Technologies used:**
    - Python & Streamlit for the interface
    - PyMuPDF for PDF text extraction
    - SentenceTransformers for embeddings
    - FAISS for semantic search
    - Hugging Face Transformers for answer generation
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 Model Information")
    
    if st.session_state.model_loaded:
        st.markdown(f'<div class="model-selector">Current Model: <b>{selected_model}</b></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">No model loaded yet</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Model options:**
    - **Small & Fast**: T5-Small - Fastest but lower quality
    - **Balanced**: DistilBERT - Good balance of speed and quality
    - **High Quality**: BERT Large - Slower but higher quality answers
    - **Advanced**: RoBERTa - High quality with better context understanding
    """)
    
    st.markdown("---")
    st.markdown("### 🗑️ Clear Session")
    if st.button("Clear All Documents and History"):
        st.session_state.pdf_texts = {}
        st.session_state.index = None
        st.session_state.embedding_model = None
        st.session_state.qa_model = None
        st.session_state.chunks = []
        st.session_state.chunk_doc_map = {}
        st.session_state.questions_answers = []
        st.session_state.model_loaded = False
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #777;'>"
    "StudyMate - AI-Powered PDF-Based Q&A System for Students<br>"
    "Built with Python, Streamlit, and Hugging Face Transformers"
    "</div>",
    unsafe_allow_html=True
)