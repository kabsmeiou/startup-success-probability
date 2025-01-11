from typing import Union

from enum import Enum # for options
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class platform_name(str, Enum):
    fongogo = "fongogo"
    kongogo = "kongogo"
    fongokongogo = "fongokongogo"
    kongokongogo = "kongokongogo"

class Startup(BaseModel):
    platform_name: str
    mass_funding_type: float
    is_offer: Union[bool, None] = None


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}