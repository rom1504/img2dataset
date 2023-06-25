# data loader calling http server to get tar files

import requests
from img2dataset.core.reader import Reader
import webdataset as wds
import fsspec
from .launcher import launcher
import uuid
from multiprocessing import get_context
import time
import os
import random
import fire
import json
from copy import deepcopy
import math
import torch

def random_uuid():
    return uuid.UUID(bytes=bytes(random.getrandbits(8) for _ in range(16)), version=4)

# https://stackoverflow.com/a/73231916

# need here
# - dataloader : multiple services
# - reader/preparer

def launch(processes_count):
    ctx = get_context("spawn")
    tmp_file_name = "/tmp/" + str(uuid.uuid4())
    p = ctx.Process(target=launcher, kwargs={"tmp_file_name": tmp_file_name, "processes_count": processes_count})
    p.start()
    while 1:
        time.sleep(0.1)
        if os.path.exists(tmp_file_name):
            break
    with open(tmp_file_name, "r", encoding="utf8") as f:
        load_balancer_url = f.read()

    time.sleep(1) # fix by doing calls to load balancer until all clients are ready
    return load_balancer_url, p


def f(start_launcher=True, processes_count=1, **kwargs):
    # call http server to generate tar files
    print("I start")

    print("allo", processes_count)

    if start_launcher:
        load_balancer_url, p = launch(processes_count=processes_count)

    print("launcher started")
        
    # instanciate reader/preparer

    tmp_path = kwargs["output_folder"] + "/_tmp"


    def make_path_absolute(path):
        fs, p = fsspec.core.url_to_fs(path)
        if fs.protocol == "file":
            return os.path.abspath(p)
        return path

    kwargs["output_folder"] = output_folder = make_path_absolute(kwargs["output_folder"])
    kwargs["url_list"] = make_path_absolute(kwargs["url_list"])


    tmp_path = output_folder + "/_tmp"
    fs, tmp_dir = fsspec.core.url_to_fs(tmp_path)
    if not fs.exists(tmp_dir):
        fs.mkdir(tmp_dir)


    reader = Reader(
        kwargs["url_list"],
        kwargs["input_format"],
        kwargs["url_col"],
        kwargs["caption_col"],
        kwargs["save_additional_columns"],
        kwargs["number_sample_per_shard"],
        set(),
        tmp_path,
    )

    print("sharding")

    result = reader.prepare(1) # this is too slow. Replace by less feather files and using partial read ?

    feather_files = [x[1] for x in result]

    def feather_to_tar(feather_file):
        t = time.time()

        input_feather = feather_file
        
        random.seed(feather_file)
        unique_id = str(random_uuid())
        output_tar_prefix = kwargs["output_folder"] + "/" +unique_id

        params = deepcopy(kwargs)
        params["input_file"] = input_feather
        params["output_file_prefix"] = output_tar_prefix

        response = requests.post(load_balancer_url + '/download', json=params)

        if response.status_code != 200:
            print("Error: ", response.status_code)
            return
        
        if json.loads(response.content) != True:
            print("Error: ", json.loads(response.content))
            return
        tar_path = f'{output_tar_prefix}.tar'

        time.sleep(0.5)

        print(f"Took {time.time() - t} seconds to download {tar_path}")

        return tar_path

    def feather_to_tar_generator(feather_file_generator):
        for feather_file in feather_file_generator:
            print("feather_file", feather_file)
            yield {"url":feather_to_tar(feather_file["url"])}
    
    def is_present(item):
        if len(item) != 2:
            return False
        if item[0] is None:
            return False
        if item[1] is None:
            return False
        return True

    def filter_out_empty(items):
        for item in items:
            if is_present(item):
                yield item

    dataset = wds.DataPipeline(
        wds.ResampledShards(feather_files),
        feather_to_tar_generator,
        wds.tarfile_to_samples(),
        #wds.shuffle(1000),
        wds.decode("torchrgb"),
        # at this point, we have an list of decompressed training samples from each shard in this worker in sequence
        wds.to_tuple("jpg", "txt"),
        filter_out_empty,
        wds.batched(16)
    )

    dl = torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=False, num_workers=processes_count, prefetch_factor=64
    )

    for image_batch, caption_batch in dl:
        yield image_batch, caption_batch

    if start_launcher:
        p.terminate()


def run_dataloader(start_launcher=True, processes_count=2, **kwargs):

    dataloader = f(start_launcher=start_launcher, processes_count=processes_count, **kwargs)
    i = 0
    t = time.time()
    for image_batch, caption_batch in dataloader:
        i+=16
        print(len(image_batch))
        print(f"sample/s = {i/(time.time()-t)}")
        time.sleep(0.08)


def main():
    fire.Fire(run_dataloader)


if __name__ == "__main__":
    main()
