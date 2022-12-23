
"""service"""

from fastapi import FastAPI
import fire
from copy import deepcopy
from ..core.downloader import download_shard
import uvicorn
import requests
from typing import List, Optional
import asyncio


from threading import Semaphore


def service(
    load_balancer_url: Optional[str] = None,
):
    """Download is the main entry point of img2dataset, it uses multiple processes and download multiple files"""


    semaphore = Semaphore(1)
    

    app = FastAPI()

    @app.get("/")
    def get():
        return "hi"

    # make it better
    @app.post("/download")
    def download(config_args: dict):
        semaphore.acquire()
        print(config_args)
        config_args["delete_input_shard"] = False
        r = download_shard(**config_args)
        semaphore.release()
        return r
    
    async def run():
        config = uvicorn.Config(app, port=0, log_level="info")
        server = uvicorn.Server(config)

        task = asyncio.create_task(server.serve())

        while not server.started:
            await asyncio.sleep(0.1)

        service_url = None

        for server in server.servers:
            for socket in server.sockets:
                service_url = f"http://{socket.getsockname()[0]}:{socket.getsockname()[1]}"
                break
        if service_url is None:
            raise Exception("Could not find service url")
        
        if load_balancer_url is not None:
            requests.get(load_balancer_url+ "/add_service", params={"service_url": service_url})
            print("added service to load balancer")

        await task

    LOOP = asyncio.get_event_loop()
    LOOP.run_until_complete(run())



def main():
    fire.Fire(service)


if __name__ == "__main__":
    main()
