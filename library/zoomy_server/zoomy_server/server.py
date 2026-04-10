from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from zoomy_server.adapter import SolverAdapter
from zoomy_server.routes import router, set_adapter


class ZoomyServer:
    def __init__(self, adapter: SolverAdapter):
        self.adapter = adapter
        self.app = FastAPI(title=f"Zoomy Solver Server ({adapter.tag})", version="1.0")
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        set_adapter(adapter)
        self.app.include_router(router)

    def run(self, host="0.0.0.0", port=8080):
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)
