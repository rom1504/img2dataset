## CC3M

CC3M is a dataset of 3 million image and caption.

### Download the tsv file

Go to https://ai.google.com/research/ConceptualCaptions/download and press download
That's a 500MB tsv file

Add the column names at the top of the file with `sed -i '1s/^/caption\turl\n/' cc3m.tsv`

### Download with img2dataset

Run this command. It will download the cc3m dataset as resized images in the webdataset format.

```
img2dataset --url_list cc3m.tsv --input_format "tsv"\
         --url_col "url" --caption_col "caption" --output_format webdataset\
           --output_folder cc3m --processes_count 16 --thread_count 64 --image_size 256\
             --enable_wandb True
```

### Benchmark

https://wandb.ai/rom1504/img2dataset/reports/Download-cc3m-with-img2dataset--VmlldzoxMjE5MTE4

This dataset has a lot of high resolution images, so this results in about 850 image downloader per second. Overall this takes about one hour. Using a computer with 16 cores, and 2Gbps of bandwidth.
