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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import numpy as np

# Set NLTK data path
nltk.data.path.append("C:\\Users\\Jay Manish Guri\\AppData\\Roaming\\nltk_data")
nltk.download('punkt')
nltk.download('stopwords')

app = FastAPI(title="PDF Assistant: Flashcard and Q&A System")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_FOLDER = Path("pdf_uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Model and Embeddings
class SentenceTransformerEmbeddings(Embeddings):
    def _init_(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# Pydantic Models
class FlashCard(BaseModel):
    topic: str
    question: str
    answer: str

class FlashCardSessionResponse(BaseModel):
    session_id: str
    flashcard: Optional[FlashCard]

class QuestionRequest(BaseModel):
    question: str
    previous_conversations: List[str] = []
    chat_history: List[str] = []

class QuestionResponse(BaseModel):
    answer: str
    sentiment: str

# Global Variables
embeddings = SentenceTransformerEmbeddings()
vector_store = InMemoryVectorStore(embeddings)
model = OllamaLLM(model="qwen2.5:7b", temperature=0.7)
sentiment_analyzer = SentimentIntensityAnalyzer()

flashcard_sessions = {}

# Templates
TEMPLATES = {
    'neutral': """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If the user asks you for a summary or explanation of a particular topic, find that topic and summarize or explain it before giving the answer.
    Previous Conversations (User-Assistant Q&A): {previous_conversations}
    Chat History (User-Assistant Dialogue): {chat_history}
    Question: {question} 
    Context: {context} 
    Answer in a neutral and straightforward manner:
    """,
    
    'positive': """
    You are an enthusiastic and encouraging assistant. Use the following pieces of retrieved context to answer the question. 
    If the user asks you for a summary or explanation of a particular topic, find that topic and summarize or explain it before giving the answer.
    Previous Conversations (User-Assistant Q&A): {previous_conversations}
    Chat History (User-Assistant Dialogue): {chat_history}
    Question: {question} 
    Context: {context} 
    Answer in an upbeat and positive tone:
    """,
    
    'negative': """
    You are an empathetic and supportive assistant. Use the following pieces of retrieved context to answer the question. 
    If the user asks you for a summary or explanation of a particular topic, find that topic and summarize or explain it before giving the answer.
    Previous Conversations (User-Assistant Q&A): {previous_conversations}
    Chat History (User-Assistant Dialogue): {chat_history}
    Question: {question} 
    Context: {context} 
    Answer in a supportive and understanding tone:
    """
}

# Flashcard Generation Templates
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

# Utility Functions
def chunks(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def get_sentiment(text: str) -> str:
    scores = sentiment_analyzer.polarity_scores(text)
    compound_score = scores['compound']
    
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

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

async def process_pdf(file_path: Path):
    try:
        loader = PDFPlumberLoader(str(file_path))
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        
        chunks = text_splitter.split_documents(documents)
        vector_store.add_documents(chunks)
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

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

def answer_question(question: str, previous_conversations: List[str], chat_history: List[str]) -> tuple[str, str]:
    try:
        sentiment = get_sentiment(question)
        template = TEMPLATES[sentiment]
        
        # Retrieve relevant documents
        documents = vector_store.similarity_search(question)
        context = "\n\n".join([doc.page_content for doc in documents])
        
        # Format conversation history
        conversation_history = "\n".join(previous_conversations)
        chat_history_str = "\n".join(chat_history)
        
        # Generate response
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        response = chain.invoke({
            "question": question,
            "previous_conversations": conversation_history,
            "chat_history": chat_history_str,
            "context": context
        })
        
        return response, sentiment
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

# API Endpoints
@app.post("/upload", response_model=FlashCardSessionResponse)
async def upload_pdf_for_flashcards(
    file: UploadFile = File(...),
    num_cards: int = Form(...),
    specific_topic: Optional[str] = Form(None)
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are supported")
    
    if not 1 <= num_cards <= 20:
        raise HTTPException(400, "Number of cards must be between 1 and 20")
    
    file_path = UPLOAD_FOLDER / f"{uuid4()}.pdf"
    with file_path.open("wb") as f:
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
        if file_path.exists():
            file_path.unlink()

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

@app.post("/ask/", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    if not vector_store.docstore._dict:  # Check if any documents have been processed
        raise HTTPException(status_code=400, detail="No documents have been processed. Please upload a PDF first.")
    
    answer, sentiment = answer_question(
        request.question,
        request.previous_conversations,
        request.chat_history
    )
    
    return QuestionResponse(answer=answer, sentiment=sentiment)

@app.get("/health/")
async def health_check():
    return {"status": "healthy"}

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)