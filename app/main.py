"""FastAPI application entry point wiring CORS middleware and feature routers."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import chat, documents, users

app = FastAPI(
    title="Multi-Tenant RAG Assistant API",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users.router)
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/")
def home():
    return {"message": "Multi-Tenant RAG Assistant API is running"}
