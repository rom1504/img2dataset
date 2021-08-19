from multiprocessing import Pool
from tqdm import tqdm
import cv2
import os
import urllib.request
import fire
import functools
import webdataset as wds
import io
import numpy as np

def process_image(row, IMAGE_SIZE, resize_mode, resize_only_if_bigger):
    key, url = row
    try:
        request = urllib.request.Request(
                url,
                data=None,
                headers={'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0'}
            )
        img_stream = io.BytesIO(urllib.request.urlopen(request, timeout=10).read())
        img = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)
        if not resize_only_if_bigger or img.shape[0] > IMAGE_SIZE or img.shape[1] > IMAGE_SIZE:
            if resize_mode == "border":
                img = resize_with_border(img, IMAGE_SIZE)
            elif resize_mode == "no":
                img = img
            elif resize_mode == "keep_ratio":
                img = resize_keep_ratio(img, IMAGE_SIZE)

        return key, img
    except Exception as _:
        return None, None

# keep the ratio, smaller side is desired_size
def resize_keep_ratio(im, desired_size=256):
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/min(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))

    return im

# resize and add a border, larger side is desired_size
def resize_with_border(im, desired_size=256):
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    return new_im

def download(
    url_list,
    image_size=256,
    output_folder='images',
    thread_count=256,
    resize_mode="border",
    resize_only_if_bigger=False,
    output_format="files"):

    IMAGE_SIZE = image_size
    IMAGE_DIR = output_folder
    if not os.path.exists(IMAGE_DIR):
        os.mkdir(IMAGE_DIR)

    images_to_dl = []
    with open(url_list, encoding='utf-8') as file:
        images_to_dl = list(enumerate(file.readlines()))

    downloader = functools.partial(process_image, IMAGE_SIZE=IMAGE_SIZE,\
         resize_mode=resize_mode, resize_only_if_bigger=resize_only_if_bigger)
    pool = Pool(thread_count)

    if output_format == "files":
        for key, img in tqdm(pool.imap_unordered(downloader, images_to_dl), total=len(images_to_dl)):
            if key is None:
                continue
            part = key // 10000
            folder = f"{IMAGE_DIR}/{part}"
            if not os.path.exists(folder):
                os.mkdir(folder)
            filename = f'{folder}/{key}.jpg'
            cv2.imwrite(filename, img)
            pass
    
    elif output_format == "webdataset":
        pattern = os.path.join(output_folder, f"%06d.tar")
        with wds.ShardWriter(pattern, maxsize=1e9, maxcount=10000) as sink:
            pool = Pool(thread_count)
            for key, img in tqdm(pool.imap_unordered(downloader, images_to_dl), total=len(images_to_dl)):
                if key is None:
                    continue


                img_str = cv2.imencode('.jpg', img)[1].tobytes()

                key = "%09d" % key

                sample = {
                    "__key__": key,
                    "jpg": img_str
                }
                sink.write(sample)


def main():
    fire.Fire(download)

if __name__ == '__main__':
    main()