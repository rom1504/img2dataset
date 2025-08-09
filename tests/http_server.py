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


@app.get("/robots.txt")
async def get_robots_txt():
    return Response(content="User-Agent: *\nDisallow: /disallowed/robots", media_type="text/plain")


app.mount("/allowed", StaticFiles(directory=test_folder), name="static_allowed")
app.mount("/disallowed/robots", StaticFiles(directory=test_folder), name="static_disallowed_robots")
app.mount("/disallowed/header", StaticFilesXRobotsTagHeader(directory=test_folder), name="static_disallowed_header")
