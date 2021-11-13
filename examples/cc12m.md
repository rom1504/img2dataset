## CC12M

[CC12M](https://github.com/google-research-datasets/conceptual-12m) is a dataset of 12 million image and caption.


### Download the metadata

`wget https://storage.googleapis.com/conceptual_12m/cc12m.tsv`
That's a 2.6GB file

Add the column names at the top of the file with `sed -i '1s/^/url\tcaption\n/' cc12m.tsv`

### Download the images with img2dataset

Run this command. It will download the cc12m dataset as resized images in the webdataset format.

```
img2dataset --url_list cc12m.tsv --input_format "tsv"\
         --url_col "url" --caption_col "caption" --output_format webdataset\
           --output_folder cc12m --processes_count 16 --thread_count 64 --image_size 256\
             --enable_wandb True
```

### Benchmark

https://wandb.ai/rom1504/img2dataset/reports/Download-cc12m-with-img2dataset--VmlldzoxMjIxMTY0
* 630 sample/s : cc12m has a lot of large images so resizing makes cpu the bottleneck
* total: 5h
* output: 331GB

