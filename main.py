from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
from cc_assistant import agent
import json

app = FastAPI(
    title="CardAdvisor API",
    description="AI-powered credit card recommendation system",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Handle chat requests from the frontend"""
    try:
        # Use the existing agent to process the request
        response = agent.invoke({"input": request.message})
        output = response.get("output", str(response))
        details = response.get("intermediate_steps", None)
        
        return ChatResponse(
            response=output,
            details=details
        )
    except Exception as e:
        return ChatResponse(
            response="",
            error=f"Sorry, something went wrong: {str(e)}"
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "CardAdvisor API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 