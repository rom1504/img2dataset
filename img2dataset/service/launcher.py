# can also be called stream distributor

# job is to start the load balancer and the service 

# multiprocessing and pyspark and maybe ssh/slurm

"""distributor defines the distribution strategies for img2dataset"""

from multiprocessing import get_context
from .service import service
from.loadbalancer import load_balancer
import fire

import time
import uuid
import os


def multiprocessing_launcher(processes_count, config_args, tmp_file_name):
    """Start N workers"""
    ctx = get_context("spawn")
    if tmp_file_name is None:
        tmp_file_name = "/tmp/" + str(uuid.uuid4())
    load_balancer_process = ctx.Process(target=load_balancer, args=(tmp_file_name,))
    load_balancer_process.start()
    while 1:
        time.sleep(0.1)
        if os.path.exists(tmp_file_name):
            break
    with open(tmp_file_name, "r", encoding="utf8") as f:
        load_balancer_url = f.read()

    # need to get the load balancer url

    processes = []
    for _ in range(processes_count):
        config_args["load_balancer_url"] = load_balancer_url
        p = ctx.Process(target=service, kwargs=config_args)
        p.start()
        processes.append(p)
    

    load_balancer_process.join()
    for p in processes:
        p.join()

def launcher(
    processes_count: int = 1,
    tmp_file_name : str = None,
    **kwargs,
    ):
    """Download is the main entry point of img2dataset, it uses multiple processes and download multiple files"""
    config_parameters = kwargs
    multiprocessing_launcher(processes_count, config_parameters, tmp_file_name)


def main():
    fire.Fire(launcher)


if __name__ == "__main__":
    main()
