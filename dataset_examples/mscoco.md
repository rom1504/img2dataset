## mscoco

[mscoco](https://academictorrents.com/details/74dec1dd21ae4994dfd9069f9cb0443eb960c962) train split is a dataset of 600 thousands image and caption. 


### Download the metadata

`wget https://huggingface.co/datasets/ChristophSchuhmann/MS_COCO_2017_URL_TEXT/resolve/main/mscoco.parquet`
That's a 18M file. It contains the train split from [mscoco](https://academictorrents.com/details/74dec1dd21ae4994dfd9069f9cb0443eb960c962)


### Download the images with img2dataset

Run this command. It will download the mscoco dataset as resized images in the webdataset format.

```
img2dataset --url_list mscoco.parquet --input_format "parquet"\
         --url_col "URL" --caption_col "TEXT" --output_format webdataset\
           --output_folder mscoco --processes_count 16 --thread_count 64 --image_size 256\
             --enable_wandb True
```

### Benchmark

https://wandb.ai/rom1504/img2dataset/reports/MSCOCO--VmlldzoxMjczMTkz
* 800 sample/s
* total: 10min
* output: 20GB

