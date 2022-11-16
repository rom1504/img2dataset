import os

from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles


class StaticFilesXRobotsTagHeader(StaticFiles):
    async def get_response(self, *args, **kwargs) -> Response:
        response = await super().get_response(*args, **kwargs)
        response.headers["X-Robots-Tag"] = "noai, noimageai, noindex, noimageindex, nofollow"
        return response


app = FastAPI()

current_folder = os.path.dirname(__file__)
test_folder = str(current_folder) + "/" + "resize_test_image"


@app.get("/")
async def get():
    return "hi"


app.mount("/allowed", StaticFiles(directory=test_folder), name="static_allowed")
app.mount("/disallowed", StaticFilesXRobotsTagHeader(directory=test_folder), name="static_disallowed")
