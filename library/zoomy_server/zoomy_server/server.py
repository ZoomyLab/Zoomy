from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from zoomy_server.adapter import SolverAdapter
from zoomy_server.routes import router, set_adapter


class ZoomyServer:
    def __init__(self, adapter: SolverAdapter):
        self.adapter = adapter
        self.app = FastAPI(title=f"Zoomy Solver Server ({adapter.tag})", version="1.0")
        # ``allow_private_network`` is what lets a browser-served GUI reach this
        # server at all.  A page served from a PUBLIC origin (the deployed GUI on
        # github.io) asking for a PRIVATE address (localhost / a LAN host) is
        # Private Network Access: Chrome sends a preflight carrying
        # ``Access-Control-Request-Private-Network: true``, and Starlette's
        # CORSMiddleware answers 400 "Disallowed CORS private-network" unless
        # this is set.  The request is then rejected before any route runs, so
        # the server log stays silent while the GUI reports "no healthy
        # zoomy-server" -- and ``curl`` still succeeds, because curl does not
        # send that header.  Note localhost is NOT the problem: Chrome treats
        # http://localhost as a potentially trustworthy origin, so ordinary
        # mixed-content blocking never applies here.
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            allow_private_network=True,
        )
        set_adapter(adapter)
        self.app.include_router(router)

    def run(self, host="0.0.0.0", port=8080):
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)
