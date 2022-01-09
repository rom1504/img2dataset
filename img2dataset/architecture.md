Img2dataset is split in these modules:

* reader: read the url data and yield it as file shards (list of arrow files)
* writer: write the image data
* resizer: take as input images, and return resized images
* downloader: takes one shard, read it to memory, resize it using the resizer, write it using the writer
* main: takes a collection of files, reads them as shards using the reader, spawn N processes and in each use a downloader to process shards

Main is the only one that is exposed to the user

The objective of this split in modules is to make it easier to expand the functionalities (new input and output format, new resizing, new ways to distribute)

