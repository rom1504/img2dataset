from multiprocessing import Pool
from tqdm import tqdm
import csv
import cv2
import os
import urllib.request
import hashlib
import fire
import functools

def process_image(row, IMAGE_SIZE):
    url, filename = row
    if os.path.exists(filename):
        return
    try:
        request = urllib.request.Request(
                url,
                data=None,
                headers={'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0'}
            )
        content = urllib.request.urlopen(request, timeout=10).read()
        with open(filename, 'wb') as outfile:
            outfile.write(content)
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        img = resize_with_border(img, IMAGE_SIZE)

        cv2.imwrite(filename, img)
    except Exception as e:
        # todo remove
        if os.path.exists(filename):
            os.remove(filename)
        pass


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

def download(url_list, image_size=256, output_folder='images', thread_count=256):
    IMAGE_SIZE = image_size
    IMAGE_FORMAT = 'jpg'

    IMAGE_DIR = output_folder
    if not os.path.exists(IMAGE_DIR):
        os.mkdir(IMAGE_DIR)


    images_to_dl = []
    with open(url_list, encoding='utf-8') as file:
        lines = file.readlines()
        for i, line in tqdm(enumerate(lines)):
            url = line
            part = i // 10000
            folder = f"{IMAGE_DIR}/{part}"
            if not os.path.exists(folder):
                os.mkdir(folder)
            filename = f'{folder}/{i}.jpg'
            images_to_dl.append((url, filename))

    images_to_dl = images_to_dl

    downloader = functools.partial(process_image, IMAGE_SIZE=IMAGE_SIZE)
    pool = Pool(thread_count)
    for _ in tqdm(pool.imap_unordered(downloader, images_to_dl), total=len(images_to_dl)):
        pass


def main():
    fire.Fire(download)

if __name__ == '__main__':
    main()