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
import exifread
import json
import glob
import logging
logging.getLogger('exifread').setLevel(level=logging.CRITICAL)

def download_image(row):
    key, url = row
    try:
        request = urllib.request.Request(
            url,
            data=None,
            headers={'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0'}
        )
        with urllib.request.urlopen(request, timeout=10) as r:
            img_stream = io.BytesIO(r.read())
        return key, img_stream, None
    except Exception as err:
        return key, None, str(err)


def resize_image(img_stream, image_size, resize_mode, resize_only_if_bigger):
    try:
        img = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)
        original_height = img.shape[0]
        original_width = img.shape[1]
        if not resize_only_if_bigger or img.shape[0] > image_size or img.shape[1] > image_size:
            if resize_mode == "no" or image_size == original_height == original_width:
                img = img
            elif resize_mode == "border":
                img = resize_with_border(img, image_size)
            elif resize_mode == "keep_ratio":
                img = resize_keep_ratio(img, image_size)
            elif resize_mode == "center_crop":
                img = resize_center_crop(img, image_size)
        height = img.shape[0]
        width = img.shape[1]
        img_str = cv2.imencode('.jpg', img)[1].tobytes()
        return img_str, width, height, original_width, original_height, None
    except Exception as err:
        return None, None, None, None, None, str(err)

# keep the ratio, smaller side is desired_size
def resize_keep_ratio(im, desired_size=256):
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/min(old_size)
    new_size = tuple([int(round(x*ratio)) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]), interpolation = cv2.INTER_LANCZOS4)

    return im

# keep the ratio, largest side center cropped so output is squared
def resize_center_crop(im, desired_size=256):
    im = resize_keep_ratio(im, desired_size)
    h, w = im.shape[:2]
    y1 = (h - desired_size) // 2
    x1 = (w - desired_size) // 2
    return im[y1:y1+desired_size, x1:x1+desired_size]

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

class WebDatasetSampleWriter:
    def __init__(self, shard_id, output_folder):
        shard_name = "%05d" % shard_id
        self.tarwriter = wds.TarWriter(f"{output_folder}/{shard_name}.tar")

    def write(self, img_str, key, caption, meta):
        key = "%09d" % key
        sample = {
            "__key__": key,
            "jpg": img_str
        }
        if caption is not None:
            sample["txt"] = caption
        if meta is not None:
            sample["json"] = json.dumps(meta, indent=4)
        self.tarwriter.write(sample)

    def close(self):
        self.tarwriter.close()

class FilesSampleWriter:
    def __init__(self, shard_id, output_folder):
        shard_name = "%05d" % shard_id
        self.subfolder = f"{output_folder}/{shard_name}"
        if not os.path.exists(self.subfolder):
            os.mkdir(self.subfolder)
    
    def write(self, img_str, key, caption, meta):
        key = "%04d" % key
        filename = f'{self.subfolder}/{key}.jpg'
        with open(filename, "wb") as f:
            f.write(img_str)
        if caption is not None:
            caption_filename = f'{self.subfolder}/{key}.txt'
            with open(caption_filename, "w") as f:
                f.write(caption)
        if meta is not None:
            j = json.dumps(meta, indent=4)
            meta_filename = f'{self.subfolder}/{key}.json'
            with open(meta_filename, "w") as f:
                f.write(j)
    
    def close(self):
        pass



def one_process_downloader(row, sample_writer_class, resizer, thread_count, save_metadata, output_folder, column_list):
    shard_id, shard_to_dl = row

    if save_metadata:
        metadatas = []
    
    total = len(shard_to_dl)
    successes = 0
    failed_to_download = 0
    failed_to_resize = 0
    url_indice = column_list.index("url")
    caption_indice = column_list.index("caption") if "caption" in column_list else None
    key_url_list = [(key, x[url_indice]) for key, x in shard_to_dl]

    sample_writer = sample_writer_class(shard_id, output_folder)
    with ThreadPool(thread_count) as thread_pool:
        for key, img_stream, error_message in thread_pool.imap_unordered(download_image, key_url_list):
            _, sample_data = shard_to_dl[key]
            meta = {
                    **{column_list[i]:sample_data[i] for i in range(len(column_list))},
                    "key": key,
                    "shard_id": shard_id,
                    "status": None,
                    "error_message": error_message,
                    "width": None,
                    "height": None,
                    "exif": None,
                }
            if error_message is not None:
                failed_to_download+=1
                status="failed_to_download"
                if save_metadata:
                    meta["status"] = status
                    metadatas.append(meta)
                continue
            img, width, height, original_width, original_height, error_message = resizer(img_stream)
            if error_message is not None:
                failed_to_resize+=1
                status="failed_to_resize"
                if save_metadata:
                    meta["status"] = status
                    metadatas.append(meta)
                continue
            successes+=1
            status="success"

            if save_metadata:
                try:
                    exif = json.dumps({k:str(v).strip()  for k, v in exifread.process_file(img_stream, details=False).items() if v is not None})
                except Exception as _:
                    exif = None
                meta["status"] = status
                meta["width"] = width
                meta["height"] = height
                meta["original_width"] = original_width
                meta["original_height"] = original_height
                meta["exif"] = exif
                metadatas.append(meta)
            else:
                meta=None

            sample_writer.write(img, key, sample_data[caption_indice] if caption_indice is not None else None, meta)
        
        sample_writer.close()
        thread_pool.terminate()
        thread_pool.join()
        del thread_pool

    if save_metadata:
        df = pd.DataFrame(metadatas)
        shard_name = "%05d" % shard_id
        df.to_parquet(output_folder+"/"+shard_name+".parquet")
    
    return (total, successes, failed_to_download, failed_to_resize)

def download(
    url_list,
    image_size=256,
    output_folder='images',
    processes_count=1,
    resize_mode="border",
    resize_only_if_bigger=False,
    output_format="files",
    input_format="txt",
    url_col="url",
    caption_col=None,
    thread_count=256,
    number_sample_per_shard=10000,
    save_metadata=True,
    save_additional_columns=None
    ):

    def download_one_file(url_list):
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
            start_shard_id=0
        else:
            existing_top_level_files = glob.glob(output_folder+"/*")
            if len(existing_top_level_files) == 0:
                start_shard_id=0
            else:
                start_shard_id=max([int(x.split("/")[-1].split(".")[0]) for x in existing_top_level_files])+1

        if input_format == "txt":
            images_to_dl = []
            with open(url_list, encoding='utf-8') as file:
                images_to_dl = [(url, ) for url in file.readlines()]
            column_list=["url"]
        elif input_format in ["csv", "tsv", "parquet"]:
            if input_format == "csv":
                df = pd.read_csv(url_list)
            elif input_format == "tsv":
                df = pd.read_table(url_list)
            elif input_format == "parquet":
                df = pd.read_parquet(url_list)
            column_list = save_additional_columns if save_additional_columns is not None else []
            df = df.rename(columns={caption_col: "caption", url_col: "url"})
            if caption_col is not None:
                column_list = column_list + ["caption", "url"]
            else:
                column_list = column_list + ["url"]
            images_to_dl = df[column_list].to_records(index=False)
            del df

        sharded_images_to_dl = []
        number_samples = len(images_to_dl)
        number_shards = math.ceil(number_samples / number_sample_per_shard)
        for shard_id in range(number_shards):
            begin_shard = shard_id * number_sample_per_shard
            end_shard = min(number_samples, (1+shard_id) * number_sample_per_shard)
            sharded_images_to_dl.append((shard_id+start_shard_id, list(enumerate(images_to_dl[begin_shard:end_shard]))))
        del images_to_dl

        if output_format == "webdataset":
            sample_writer_class = WebDatasetSampleWriter
        elif output_format == "files":
            sample_writer_class = FilesSampleWriter
        
        resizer = functools.partial(resize_image, image_size=image_size, resize_mode=resize_mode, resize_only_if_bigger=resize_only_if_bigger)
        downloader = functools.partial(one_process_downloader, sample_writer_class=sample_writer_class, resizer=resizer, \
            thread_count=thread_count, save_metadata=save_metadata, output_folder=output_folder, column_list=column_list)

        total_total = 0
        total_success = 0
        total_failed_to_download = 0
        total_failed_to_resize = 0
        with Pool(processes_count, maxtasksperchild=5) as process_pool:
            for total, successes, failed_to_download, failed_to_resize in tqdm(process_pool.imap_unordered(downloader, sharded_images_to_dl),\
                total=len(sharded_images_to_dl)):
                total_total+=total 
                total_success+=successes
                total_failed_to_download+=failed_to_download
                total_failed_to_resize+=failed_to_resize
                message=f"success={1.0*total_success/total_total:.2f} "
                message+=f"failed download={1.0*total_failed_to_download/total_total:.2f} "
                message+=f"failed resize={1.0*total_failed_to_resize/total_total:.2f}"
                tqdm.print(message+"\n" , sep=' ', end='', flush=True)
                pass
            process_pool.terminate()
            process_pool.join()
            del process_pool

    if os.path.isdir(url_list):
        input_files = glob.glob(url_list+"/*."+input_format)
    else:
        input_files = [url_list]
    
    for i, input_file in enumerate(input_files):
        print("Downloading file number "+str(i+1) +" of "+str(len(input_files))+" called "+input_file)
        download_one_file(input_file)


def main():
    fire.Fire(download)

if __name__ == '__main__':
    main()
