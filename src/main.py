from typing import Optional
from dotenv import load_dotenv
import logging

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from src.ai.graph import corp_workflow
from src.ai.utils import collect_graph_states
from src.ai.config import prepare_query_inputs, prepare_config

# Carregar variáveis de ambiente
load_dotenv()

# Configuração de log
logging.basicConfig(level=logging.INFO)

# Inicialização do FastAPI
app = FastAPI()

# Middleware para logar todas as requisições HTTP
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logging.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logging.info(f"Response status: {response.status_code}")
    return response

# Tratamento global de exceções
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred."}
    )

# Endpoint de saúde
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Endpoint principal de consulta
@app.post("/query")
async def query(request: Request):
    try:
        data = await request.json()
        inputs = prepare_query_inputs(data)
        config = prepare_config(data)

        result = collect_graph_states(corp_workflow, inputs, config=config)

        # Coleta os estados do gráfico com os inputs e configurações
        return result
    except Exception as exc:

        logging.error(f"Error during processing: {exc}")
        raise HTTPException(status_code=400, detail="Invalid input or processing error")
