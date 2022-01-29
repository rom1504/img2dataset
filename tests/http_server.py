import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


app = FastAPI()

current_folder = os.path.dirname(__file__)
test_folder = str(current_folder) + "/" + "resize_test_image"


@app.get("/")
async def get():
    return "hi"


app.mount("/", StaticFiles(directory=test_folder), name="static")
