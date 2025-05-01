from typing import Optional
from dotenv import load_dotenv
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.ai.agents.supervisor import supervisor
from src.ai.utils import collect_graph_states

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = FastAPI()

@app.middleware("http")
async def log_requests(request, call_next):
    logging.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logging.info(f"Response status: {response.status_code}")
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logging.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred."}
    )

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/query")
async def query(request: Request):
    data = await request.json()
    user_input = data.get("message", "")
    inputs = {"messages": [
        { 
            "role": "user",  
            "content": user_input
        }
    ]}
    user_id = data.get("user_id", None)
    config = {
        "stream_mode": "values",
        "stream": True,
        "stream_interval": 0.1,
        "max_tokens": 100,
        "temperature": 0.7,
        "configurable": {
            "user_id": user_id,
            "thread_id": "thread-1"
        }
    }

    if user_id:
        inputs["user_id"] = user_id
    else:
        inputs["user_id"] = "default_user"
    if "thread_id" in data:
        inputs["thread_id"] = data["thread_id"]
    else:
        inputs["thread_id"] = "thread-1"

    return collect_graph_states(graph, inputs, config=config)