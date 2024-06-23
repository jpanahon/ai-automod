"""API Endpoints for the discord bot to access"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from interface import ModelInterface

app = FastAPI()
interface = ModelInterface()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins="127.0.0.1/*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    input: str

@app.post("/")
async def get_response(input_data: InputData):
    """Get a response to a query"""
    query = input_data.input
    return {"response": float(interface.analyze_message(query)[1])}
