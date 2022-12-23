# data loader calling http server to get tar files

import requests
import webdataset as wds
from .launcher import launcher
import uuid
from multiprocessing import get_context
import time
import os
import fire
import json


# https://stackoverflow.com/a/73231916

def f(start_launcher=True, **kwargs):
    # call http server to generate tar files

    if start_launcher:
        ctx = get_context("spawn")
        tmp_file_name = "/tmp/" + str(uuid.uuid4())
        kwargs["tmp_file_name"] = tmp_file_name
        p = ctx.Process(target=launcher, kwargs=kwargs)
        p.start()
        while 1:
            time.sleep(0.1)
            if os.path.exists(tmp_file_name):
                break
        with open(tmp_file_name, "r", encoding="utf8") as f:
            load_balancer_url = f.read()


    input_feather = "/media/hd/testing/tmp_test/_tmp/211.feather"
    output_tar_prefix = "/media/hd/testing/tmp_test/211"

    params = {
        'input_file': input_feather,
        'output_file_prefix': output_tar_prefix,
    }

    response = requests.get(load_balancer_url + '/download', params=params)
    print("allo")

    if response.status_code != 200:
        print("Error: ", response.status_code)
        return
    
    print(json.loads(response.content))
    if json.loads(response.content) != True:
        print("Error: ", json.loads(response.content))
        return

    print("hi")
    tar_path = f'{output_tar_prefix}.tar'

    ds = wds.WebDataset(tar_path).decode("pil").to_tuple("jpg", "txt")

    for image, caption in ds:
        yield image, caption

    if start_launcher:
        p.terminate()


def run_dataloader(**kwargs):

    dataloader = f(start_launcher=True, **kwargs)
    for image, caption in dataloader:
        print(image, caption)


def main():
    fire.Fire(run_dataloader)


if __name__ == "__main__":
    main()
