from fastapi import FastAPI
from routes import document_routes,chat
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <- Allow all origins
    allow_credentials=False,
    allow_methods=["*"],  # <- Allow all HTTP methods
    allow_headers=["*"],  # <- Allow all headers
)

app.include_router(document_routes.router,prefix="/api/v1")
app.include_router(chat.router,prefix="/api/v1")
