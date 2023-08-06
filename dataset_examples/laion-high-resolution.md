## Laion-high-resolution

Laion high resolution is a >= 1024x1024 subset of laion5B.
It is available at https://huggingface.co/datasets/laion/laion-high-resolution
It has 170M samples

A good use case is to train a superresolution model.

### Download the metadata

Download from https://huggingface.co/datasets/laion/laion-high-resolution

```
mkdir laion-high-resolution && cd laion-high-resolution
for i in {00000..00127}; do wget https://huggingface.co/datasets/laion/laion-high-resolution/resolve/main/part-$i-5d6701c4-b238-4c0a-84e4-fe8e9daea963-c000.snappy.parquet; done
cd ..
```

### Download the images with img2dataset

```
img2dataset --url_list laion-high-resolution --input_format "parquet"\
         --url_col "URL" --caption_col "TEXT" --output_format webdataset\
           --output_folder laion-high-resolution-output --processes_count 16 --thread_count 64 --image_size 1024\
            --resize_only_if_bigger=True --resize_mode="keep_ratio" --skip_reencode=True \
             --save_additional_columns '["similarity","hash","punsafe","pwatermark","LANGUAGE"]' --enable_wandb True
```

### Benchmark

https://wandb.ai/rom1504/img2dataset/reports/laion-high-resolution--VmlldzoxOTY0MzA4

This can be downloaded at 280 sample/s so it takes 7 days to download with one 32 cores 2Gbps machine.
The result is 50TB (high resolution images are big and slow to download!)
