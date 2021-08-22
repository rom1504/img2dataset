from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import cv2
import os
import urllib.request
import fire
import functools
import webdataset as wds
import io
import numpy as np
import pandas as pd
import math


def download_image(row):
    key, caption, url = row
    try:
        request = urllib.request.Request(
            url,
            data=None,
            headers={'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0'}
        )
        img_stream = io.BytesIO(urllib.request.urlopen(request, timeout=10).read())
        return key, caption, img_stream, url
    except Exception as err:
        return None, None, None, None


def resize_image(img_stream, image_size, resize_mode, resize_only_if_bigger):
    try:
        img = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)
        if not resize_only_if_bigger or img.shape[0] > image_size or img.shape[1] > image_size:
            if resize_mode == "border":
                img = resize_with_border(img, image_size)
            elif resize_mode == "no":
                img = img
            elif resize_mode == "keep_ratio":
                img = resize_keep_ratio(img, image_size)
        img_str = cv2.imencode('.jpg', img)[1].tobytes()
        return img_str
    except Exception as _:
        return None

# keep the ratio, smaller side is desired_size
def resize_keep_ratio(im, desired_size=256):
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/min(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]), interpolation = cv2.INTER_LANCZOS4)

    return im

# resize and add a border, larger side is desired_size
def resize_with_border(im, desired_size=256):
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]), interpolation = cv2.INTER_LANCZOS4)

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    return new_im

def webdataset_sample_writer_builder(shard_id, output_folder):
    shard_name = "%05d" % shard_id
    tarwriter = wds.TarWriter(f"{output_folder}/{shard_name}.tar")
    return functools.partial(webdataset_sample_writer, tarwriter=tarwriter)

def webdataset_sample_writer(img_str, key, caption, url, tarwriter):
    key = "%09d" % key # use sample count to determine the right 09
    sample = {
        "__key__": key,
        "jpg": img_str
    }
    if caption is not None:
        sample["txt"] = caption
    tarwriter.write(sample)

def files_sample_writer_builder(shard_id, output_folder):
    shard_name = "%05d" % shard_id
    subfolder = f"{output_folder}/{shard_name}"
    if not os.path.exists(subfolder):
        os.mkdir(subfolder)
    return functools.partial(files_sample_writer, output_folder=subfolder)

def files_sample_writer(img_str, key, caption, url, output_folder):
    filename = f'{output_folder}/{key}.jpg'
    with open(filename, "wb") as f:
        f.write(img_str)
    if caption is not None:
        caption_filename = f'{output_folder}/{key}.txt'
        with open(caption_filename, "w") as f:
            f.write(caption)

def one_process_downloader(row, sample_writer_builder, resizer, thread_count, display_processus_bar):
    shard_id, shard_to_dl = row
    sample_writer = sample_writer_builder(shard_id)
    good=0
    fail_download=0
    fail_resize=0
    done=0
    with ThreadPool(thread_count) as thread_pool:
        iterator = thread_pool.imap_unordered(download_image, shard_to_dl)
        if display_processus_bar:
            iterator = tqdm(iterator, total=len(shard_to_dl))
        for key, caption, img_stream, url in iterator:
            if (done+1) % 3000 == 0:
                print("good", 1.0*good/done)
                print("failed download", 1.0*fail_download/done)
                print("failed resize", 1.0*fail_resize/done)
            done+=1
            if key is None:
                fail_download+=1
                continue
            img = resizer(img_stream)
            if img is None:
                fail_resize+=1
                continue
            good+=1
            sample_writer(img, key, caption, url)
    return
               
def download(
    url_list,
    image_size=256,
    output_folder='images',
    processes_count=16,
    resize_mode="border",
    resize_only_if_bigger=False,
    output_format="files",
    input_format="txt",
    url_col="url",
    caption_col=None,
    thread_count=16,
    number_sample_per_shard=10000,
    display_processus_bar=False,
    ):

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    if input_format == "txt":
        images_to_dl = []
        with open(url_list, encoding='utf-8') as file:
            images_to_dl = [(i, None, url) for i, url in enumerate(file.readlines())]
    elif input_format == "csv" or input_format=="parquet":
        if input_format == "csv":
            df = pd.read_csv(url_list)
        elif input_format == "parquet":
            df = pd.read_parquet(url_list)
        if caption_col is not None:
            images_to_dl = [(i, caption, url) for i, (caption, url) in enumerate(zip(df[caption_col], df[url_col]))]
        else:
            images_to_dl = [(i, None, url) for i, url in enumerate(df[url_col])]

    sharded_images_to_dl = []
    number_samples = len(images_to_dl)
    number_shards = math.ceil(number_samples / number_sample_per_shard)
    for shard_id in range(number_shards):
        begin_shard = shard_id * number_sample_per_shard
        end_shard = min(number_samples, (1+shard_id) * number_sample_per_shard)
        sharded_images_to_dl.append((shard_id, images_to_dl[begin_shard:end_shard]))

    if output_format == "webdataset":
        sample_writer_builder = functools.partial(webdataset_sample_writer_builder, output_folder=output_folder)    
    elif output_format == "files":
        sample_writer_builder = functools.partial(files_sample_writer_builder, output_folder=output_folder)
    
    resizer = functools.partial(resize_image, image_size=image_size, resize_mode=resize_mode, resize_only_if_bigger=resize_only_if_bigger)
    downloader = functools.partial(one_process_downloader, sample_writer_builder=sample_writer_builder, resizer=resizer, \
        thread_count=thread_count, display_processus_bar=display_processus_bar)

    with Pool(processes_count, maxtasksperchild=10) as process_pool:
        for _ in tqdm(process_pool.imap_unordered(downloader, sharded_images_to_dl), total=len(sharded_images_to_dl)):
            pass


def main():
    fire.Fire(download)

if __name__ == '__main__':
    main()
