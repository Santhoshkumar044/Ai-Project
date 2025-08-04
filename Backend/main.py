from fastapi import FastAPI
from routes import document_routes

app = FastAPI()

app.include_router(document_routes.router)
