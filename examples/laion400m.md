## laion-400m

[laion-400m](https://laion.ai/laion-400-open-dataset/) is a 400M image text dataset

### Download the metadata

```
wget -l1 -r --no-parent https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/
mv the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/ .
```

### Download the images with img2dataset

```
img2dataset --url_list laion400m-meta --input_format "parquet"\
         --url_col "URL" --caption_col "TEXT" --output_format webdataset\
           --output_folder laion400m-data --processes_count 16 --thread_count 128 --image_size 256\
             --save_additional_columns '["NSFW","similarity","LICENSE"]' --enable_wandb True
```

### Benchmark

This can be downloaded at 1300 sample/s so it takes 3.5 days to download with one 16 cores 2Gbps machine.
The result is 10TB
