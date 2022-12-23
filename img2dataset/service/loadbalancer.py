
"""service"""

from fastapi import FastAPI
import fire
import uvicorn
import asyncio
from typing import List
import time
import collections
import requests
import fire
import aiohttp

import threading
import asyncio


class ServiceClient:
    def __init__(self, service_url):
        self.service_url = service_url
        self.available = True

    def is_available(self):
        return self.available

    async def download(self, input_file, output_file_prefix):
        self.available = False
        # call the service on /download with input_file and output_file_prefix using requests
        
        print(f"service {self.service_url} is downloading {input_file} to {output_file_prefix}")
        async with aiohttp.ClientSession() as session:
            async with session.get(self.service_url+"/download", params={"input_file": input_file, "output_file_prefix": output_file_prefix}) as resp:
                print(resp.status)
                print(await resp.text())
        self.available = True
        return True

# load balancer has a dict of services, a shared queue, a unqueue thread and an download method
# the unqueue thread will try to call a service that is available, and loop to wait
class LoadBalancer:
    def __init__(self):
        self.services = []
        self.queue = collections.deque([])
        loop = asyncio.get_event_loop()
        self.task = loop.create_task(self.unqueue())
        self.done_shard = dict()

    def add_service(self, service_url):
        self.services.append(ServiceClient(service_url))

    async def download(self, input_path, output_path):
        # add to queue then wait for the output_path to be available
        self.queue.append((input_path, output_path))
        while output_path not in self.done_shard:
            await asyncio.sleep(1)
        return True

    async def unqueue(self):
        loop = asyncio.get_event_loop()
        while True:
            if len(self.queue) == 0:
                await asyncio.sleep(1)
                continue
            input_path, output_path = self.queue.popleft()
            executed = False
            for service in self.services:
                if service.is_available():
                    async def f():
                        await service.download(input_path, output_path)
                        self.done_shard[output_path] = True
                    loop.create_task(f())
                    executed = True
                    break
            if not executed:
                self.queue.appendleft((input_path, output_path))
                await asyncio.sleep(1)

def load_balancer(url_output_path: str = "/tmp/load_balancer"):
    """Load balancer"""

    load_balancer = LoadBalancer()

    app = FastAPI()

    @app.get("/")
    def get():
        return "hi"

    @app.get("/download")
    async def download(input_file, output_file_prefix):
        return await load_balancer.download(input_file, output_file_prefix)

    @app.get("/add_service")
    def add_service(service_url):
        load_balancer.add_service(service_url)
        return True

    @app.get("/list_services")
    def list_services():
        return [service.service_url for service in load_balancer.services]


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

        with open(url_output_path, "w", encoding="utf8") as f:
            f.write(service_url)

        await task

    LOOP = asyncio.get_event_loop()
    LOOP.run_until_complete(run())



def main():
    fire.Fire(load_balancer)


if __name__ == "__main__":
    main()
