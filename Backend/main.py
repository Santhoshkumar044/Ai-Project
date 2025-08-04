from fastapi import FastAPI
from routes import document_routes,chat

app = FastAPI()

app.include_router(document_routes.router)
app.include_router(chat.router)

