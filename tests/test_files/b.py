
from pycurl import Curl
import pycurl
from io import BytesIO
import time

def download_image(row, timeout):
    """Download an image with urllib"""
    key, url = row
    user_agent_string = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"

    try:
        mycurl=Curl()
        mycurl.setopt(pycurl.SSL_VERIFYPEER, 0)
        mycurl.setopt(pycurl.SSL_VERIFYHOST, 0)
        mycurl.setopt(pycurl.TIMEOUT, timeout)
        mycurl.setopt(pycurl.URL, url)
        body = BytesIO()
        mycurl.setopt(pycurl.WRITEFUNCTION, body.write)
        mycurl.setopt(pycurl.USERAGENT, user_agent_string)
        mycurl.perform()
        val = body.getvalue()
        body.close()
        return key, val, None
    except Exception as e:
        return key, None, str(e)

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

for item in example_urls:
    s = time.time()
    a = download_image(item, 2)
    print(time.time() - s)
