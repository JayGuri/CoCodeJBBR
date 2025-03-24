from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import shutil
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from pathlib import Path

app = FastAPI(title="PDF QA System API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
PDFS_DIRECTORY = Path("pdf_uploads")
PDFS_DIRECTORY.mkdir(exist_ok=True)

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text):
        embedding = self.model.encode([text])[0]
        return embedding.tolist()

# Initialize components
embeddings = SentenceTransformerEmbeddings()
vector_store = InMemoryVectorStore(embeddings)
model = OllamaLLM(model="qwen2.5:7b", temperature=0.7)
sentiment_analyzer = SentimentIntensityAnalyzer()

# Templates
TEMPLATES = {
    'neutral': """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If the user asks you for a summary or explanation of a particular topic, find that topic and summarize or explain it before giving the answer.
    Previous Conversations (User-Assistant Q&A): {previous_conversations}
    Chat History (User-Assistant Dialogue): {chat_history}
    Question: {question} 
    Context: {context} 
    Answer in a neutral and straightforward manner:
    """,
    
    'positive': """
    You are an enthusiastic and encouraging assistant. Use the following pieces of retrieved context to answer the question. If the user asks you for a summary or explanation of a particular topic, find that topic and summarize or explain it before giving the answer.
    Previous Conversations (User-Assistant Q&A): {previous_conversations}
    Chat History (User-Assistant Dialogue): {chat_history}
    Question: {question} 
    Context: {context} 
    Answer in an upbeat and positive tone:
    """,
    
    'negative': """
    You are an empathetic and supportive assistant. Use the following pieces of retrieved context to answer the question. If the user asks you for a summary or explanation of a particular topic, find that topic and summarize or explain it before giving the answer.
    Previous Conversations (User-Assistant Q&A): {previous_conversations}
    Chat History (User-Assistant Dialogue): {chat_history}
    Question: {question} 
    Context: {context} 
    Answer in a supportive and understanding tone:
    """
}

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    previous_conversations: List[str] = []
    chat_history: List[str] = []

class QuestionResponse(BaseModel):
    answer: str
    sentiment: str

def get_sentiment(text: str) -> str:
    scores = sentiment_analyzer.polarity_scores(text)
    compound_score = scores['compound']
    
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

def process_pdf(file_path: Path):
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
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

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

# API endpoints
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_path = PDFS_DIRECTORY / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        success = process_pdf(file_path)
        if success:
            return {"message": "PDF processed successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to process PDF")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)