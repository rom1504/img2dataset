import urllib.request
import time
import io
import os
from threading import Thread
import queue

example_urls = [
(12, 'http://www.herteldenbirname.com/wp-content/uploads/2014/05/Italia-Independent-Flocked-Aviator-Sunglasses-150x150.jpg'),
(124, 'http://image.rakuten.co.jp/sneak/cabinet/shoes-03/cr-ucrocs5-a.jpg?_ex=128x128'),
(146, 'http://www.slicingupeyeballs.com/wp-content/uploads/2009/05/stoneroses452.jpg'),
(122, 'https://media.mwcradio.com/mimesis/2013-03/01/2013-03-01T153415Z_1_CBRE920179600_RTROPTP_3_TECH-US-GERMANY-EREADER_JPG_475x310_q85.jpg'),
(282, 'https://8d1aee3bcc.site.internapcdn.net/00/images/media/5/5cfb2eba8f1f6244c6f7e261b9320a90-1.jpg'),
(298, 'https://my-furniture.com.au/media/catalog/product/cache/1/small_image/295x295/9df78eab33525d08d6e5fb8d27136e95/a/u/au0019-stool-01.jpg'),
(300, 'http://images.tastespotting.com/thumbnails/889506.jpg'),
(330, 'https://www.infoworld.pk/wp-content/uploads/2016/02/Cool-HD-Valentines-Day-Wallpapers-480x300.jpeg'),
(361, 'http://pendantscarf.com/image/cache/data/necklace/JW0013-(2)-150x150.jpg'),
(408, 'https://www.solidrop.net/photo-6/animorphia-coloring-books-for-adults-children-drawing-book-secret-garden-style-relieve-stress-graffiti-painting-book.jpg'),
]

def download_image(row, timeout):
    """Download an image with urllib"""
    key, url = row
    img_stream = None
    user_agent_string = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
    try:
        request = urllib.request.Request(url, data=None, headers={"User-Agent": user_agent_string})
        with urllib.request.urlopen(request, timeout=timeout) as r:
            img_stream = io.BytesIO(r.read())
        return key, img_stream, None
    except Exception as err:  # pylint: disable=broad-except
        if img_stream is not None:
            img_stream.close()
        return key, None, str(err)



def sequential():
    for example_url in example_urls:
        start_time = time.time()
        download_image(example_url, 2)
        t = time.time() - start_time
        print(t)

def sequential_threads():
    for example_url in example_urls:
        start_time = time.time()
        t = Thread(target=download_image, args=[example_url, 2])
        t.start()
        t.join(0)
        ti = time.time() - start_time
        print(ti)


results = []
def download_store_image(row, timeout):
    t = time.time()
    r = download_image(row, timeout)
    if time.time() - t < timeout:
        results.append(r)

def parallel_threads():
    start_time = time.time()
    threads = []
    for example_url in example_urls:
        t = Thread(target=download_store_image, args=[example_url, 2])
        t.start()
        threads.append((t, start_time))
    for t, start_time in threads:
        t.join(max(0, 2-(time.time() - start_time)))
    ti = time.time() - start_time
    print(ti)
    print(results)

    # need to autostart threads, and wait for them to
    return
    #os._exit(0)

from collections import deque

from img2dataset.good_bad_pool import GoodBadPool

def generator():
    j = 0
    while True:
        for _, u in example_urls:
            yield (j, u)
            j+=1

def good_bad_pools_start():
    start_time = time.time()
    d = lambda row: download_image(row, 2)
    l = []
    g = generator()
    for i in range(1000):
        l.append(next(g))
    good_bad_pool = GoodBadPool(generator=iter(l), runner=d, timeout=2, pool_size=128, out_queue_max=1000)
    i = 0
    res = []
    for a in good_bad_pool.run():
        print(a)
        i+=1
        print(1.0 * i / (time.time() - start_time))
        res.append(a)
        pass
    print(sorted(res, key=lambda x: x[0]))
    print(len(res))
    print(time.time() - start_time)


# wait join, don't wait for too slow

def main():
    good_bad_pools_start()

if __name__ == "__main__":
    main()
    print("done")
    os._exit(0)
