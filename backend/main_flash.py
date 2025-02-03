import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import io
import shutil
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from pathlib import Path

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import nltk
import numpy as np

# Set NLTK data path before downloads
nltk.data.path.append("C:\\Users\\Jay Manish Guri\\AppData\\Roaming\\nltk_data")
nltk.download('punkt')
nltk.download('stopwords')

app = FastAPI(title="Flashcard Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

flashcard_sessions = {}

class FlashCard(BaseModel):
    topic: str
    question: str
    answer: str

class FlashCardSessionResponse(BaseModel):
    session_id: str
    flashcard: Optional[FlashCard]

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

embeddings = SentenceTransformerEmbeddings()
vector_store = InMemoryVectorStore(embeddings)
model = OllamaLLM(model="qwen2.5:7b", temperature=0.7)

topic_specific_template = """
Create a flashcard about {specific_topic} from the following text. Focus specifically on this topic.

Context: {context}

Generate a flashcard in this format:
Q: [Question about {specific_topic}]
A: [Clear, concise answer about {specific_topic}]
"""

general_topic_template = """
Create a flashcard about an important concept from this text.

Context: {context}

Generate a flashcard in this format:
Q: [Question that tests understanding]
A: [Clear, concise answer]
"""

UPLOAD_FOLDER = "uploaded_pdfs"
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

def chunks(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

async def process_pdf(file_path: str):
    try:
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
        if not documents:
            raise HTTPException(400, "No text extracted from PDF.")
        return documents
    except Exception as e:
        raise HTTPException(400, f"Error processing PDF: {str(e)}")

async def generate_flashcards(documents, num_cards: int, specific_topic: Optional[str] = None):
    """Generate flashcards with optional topic focus."""
    flashcards = []
    full_text = " ".join([doc.page_content for doc in documents])
    text_chunks = chunks(full_text)
    
    if specific_topic:
        prompt = ChatPromptTemplate.from_template(topic_specific_template)
        
        for chunk in text_chunks:
            if len(flashcards) >= num_cards:
                break
                
            chain = prompt | model
            response = chain.invoke({
                "specific_topic": specific_topic,
                "context": chunk
            })
            
            flashcard = parse_flashcard_response(response, specific_topic)
            if flashcard:
                flashcards.append(flashcard)
    else:
        prompt = ChatPromptTemplate.from_template(general_topic_template)
        
        for chunk in text_chunks:
            if len(flashcards) >= num_cards:
                break
                
            chain = prompt | model
            response = chain.invoke({"context": chunk})
            flashcard = parse_flashcard_response(response)
            if flashcard:
                flashcards.append(flashcard)
    
    return flashcards[:num_cards]

def parse_flashcard_response(response: str, topic: str = "General") -> Optional[FlashCard]:
    """Parse LLM response into a structured flashcard."""
    lines = response.split('\n')
    question, answer = "", ""
    
    for line in lines:
        if line.startswith('Q:'):
            question = line[2:].strip()
        elif line.startswith('A:'):
            answer = line[2:].strip()
    
    return FlashCard(topic=topic, question=question, answer=answer) if question and answer else None

@app.post("/upload", response_model=FlashCardSessionResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    num_cards: int = Form(...),
    specific_topic: Optional[str] = Form(None)
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are supported")
    
    if not 1 <= num_cards <= 20:
        raise HTTPException(400, "Number of cards must be between 1 and 20")
    
    file_path = os.path.join(UPLOAD_FOLDER, f"{uuid4()}.pdf")
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        documents = await process_pdf(file_path)
        flashcards = await generate_flashcards(documents, num_cards, specific_topic)
        
        if not flashcards:
            raise HTTPException(400, "Could not generate flashcards from the provided PDF")
        
        session_id = str(uuid4())
        flashcard_sessions[session_id] = flashcards
        
        return FlashCardSessionResponse(
            session_id=session_id,
            flashcard=flashcards[0] if flashcards else None
        )
    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/next-flashcard", response_model=FlashCardSessionResponse)
async def get_next_flashcard(request_data: dict):
    session_id = request_data.get("session_id")
    if not session_id:
        raise HTTPException(400, "Session ID is required")
        
    if session_id not in flashcard_sessions or not flashcard_sessions[session_id]:
        raise HTTPException(404, "No more flashcards available")
    
    return FlashCardSessionResponse(
        session_id=session_id,
        flashcard=flashcard_sessions[session_id].pop(0)
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)