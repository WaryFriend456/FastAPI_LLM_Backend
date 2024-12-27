from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import datetime
import uuid
from pymongo import MongoClient
from fastapi.templating import Jinja2Templates

from .database import init_db, delete_empty_sessions, get_chats_collection, get_user_collection
from .models import init_models, generate_answer, Retrival_Augmentation
from .routes import register_routes

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing models and vector database... from app.main")
    init_db()
    init_models()
    yield
    print("Cleaning up resources...")

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
init_db()

# Register routes
register_routes(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)