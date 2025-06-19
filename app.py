import os
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request


app = FastAPI()

# ✅ Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Setup templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Configure logging
logging.basicConfig(level=logging.INFO)


# Create FastAPI app
app = FastAPI(title="Voice Chat AI", description="AI voice chat application with Groq integration")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Initialize Groq client
groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
if not groq_api_key:
    logging.error("GROQ_API_KEY is not set or is empty")
    groq_client = None
else:
    groq_client = Groq(api_key=groq_api_key)

# Pydantic models
class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    success: bool

class ErrorResponse(BaseModel):
    error: str
    success: bool

class HealthResponse(BaseModel):
    status: str
    groq_configured: bool

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat", response_model=ChatResponse)
async def chat(chat_message: ChatMessage):
    """Handle chat requests and return AI responses"""
    try:
        user_message = chat_message.message.strip()
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Check if Groq client is available
        if not groq_client:
            raise HTTPException(status_code=500, detail="Groq API key not configured or invalid")
        
        # Create chat completion with Groq
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant. Provide concise, friendly responses suitable for voice conversation. Keep responses conversational and not too long."
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=150,
                timeout=30
            )
            
            ai_response = chat_completion.choices[0].message.content or "Sorry, I couldn't generate a response."
            
        except Exception as groq_error:
            logging.error(f"Groq API error: {str(groq_error)}")
            if "503" in str(groq_error) or "Service Unavailable" in str(groq_error):
                raise HTTPException(
                    status_code=503, 
                    detail="AI service is temporarily unavailable. Please try again in a moment."
                )
            else:
                raise HTTPException(
                    status_code=500, 
                    detail=f"AI service error: {str(groq_error)}"
                )
        
        return ChatResponse(response=ai_response, success=True)
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get AI response: {str(e)}"
        )

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        groq_configured=bool(groq_client)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
