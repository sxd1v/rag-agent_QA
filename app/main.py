from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Naive RAG API")

app.include_router(router)