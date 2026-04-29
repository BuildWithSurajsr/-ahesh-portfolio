from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import json
from pathlib import Path
from datetime import datetime
import os

from app.src.config import Config
from app.src.embedding import EmbeddingManager
from app.src.chat import ChatManager

app = FastAPI(title="Mahesh Rajendra Portfolio")

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates
templates = Jinja2Templates(directory="app/templates")

# Data storage
DATA_DIR = Path("data")
PROJECTS_FILE = DATA_DIR / "projects.json"
MESSAGES_FILE = DATA_DIR / "messages.json"

# Chatbot components
embedding_manager = None
chat_manager = None


@app.on_event("startup")
async def startup_event():
    global embedding_manager, chat_manager
    if Config.is_valid():
        try:
            embedding_manager = EmbeddingManager()
            chat_manager = ChatManager(Config.GOOGLE_API_KEY)
            print("Chatbot initialized successfully")
        except Exception as e:
            print(f"Chatbot init failed (will work without it): {e}")
    else:
        print("GEMINI_API_KEY not set, chatbot disabled")


class ContactMessage(BaseModel):
    name: str
    email: EmailStr
    message: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []

def load_projects():
    """Load projects from JSON file"""
    if PROJECTS_FILE.exists():
        with open(PROJECTS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_message(message_data):
    """Save contact message to JSON file"""
    messages = []
    if MESSAGES_FILE.exists():
        with open(MESSAGES_FILE, 'r') as f:
            messages = json.load(f)
    
    message_data['timestamp'] = datetime.now().isoformat()
    messages.append(message_data)
    
    with open(MESSAGES_FILE, 'w') as f:
        json.dump(messages, f, indent=2)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render home page"""
    projects = load_projects()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "projects": projects}
    )

@app.get("/project/{project_id}", response_class=HTMLResponse)
async def project_detail(request: Request, project_id: int):
    """Render project detail page"""
    projects = load_projects()
    project = next((p for p in projects if p['id'] == project_id), None)
    
    if not project:
        return templates.TemplateResponse(
            "404.html",
            {"request": request},
            status_code=404
        )
    
    return templates.TemplateResponse(
        "project.html",
        {"request": request, "project": project}
    )

@app.post("/api/contact")
async def submit_contact(name: str = Form(...), email: str = Form(...), message: str = Form(...)):
    """Handle contact form submission"""
    try:
        contact_data = {
            "name": name,
            "email": email,
            "message": message
        }
        save_message(contact_data)
        return JSONResponse(
            {"success": True, "message": "Thank you! I'll get back to you soon."}
        )
    except Exception as e:
        return JSONResponse(
            {"success": False, "message": str(e)},
            status_code=500
        )

@app.get("/api/projects")
async def get_projects():
    """API endpoint for projects"""
    return load_projects()

@app.post("/api/chat")
def chat(request: ChatRequest):
    if not embedding_manager or not chat_manager:
        return JSONResponse(
            {"answer": "Chat is currently unavailable."},
            status_code=503
        )

    try:
        relevant_docs = embedding_manager.search(request.message)
        response = chat_manager.generate_response(
            request.message, relevant_docs, request.history
        )
        return JSONResponse(response)
    except Exception as e:
        return JSONResponse(
            {"answer": "Something went wrong. Please try again."},
            status_code=500
        )


@app.get("/health")
async def health():
    return {"status": "healthy", "chatbot": bool(chat_manager)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
