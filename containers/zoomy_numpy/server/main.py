import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.api.routes import router
from server.registry import init_registry

SERVER_TAG = os.environ.get("ZOOMY_SERVER_TAG", "numpy")

app = FastAPI(title="Zoomy Solver Server", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def startup():
    init_registry()
