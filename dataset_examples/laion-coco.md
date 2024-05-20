## LAION-COCO

LAION-COCO is a 600M subset of LAION2B-EN, captioned with an ensemble of BLIP L/14 and 2 CLIP versions (L/14 and RN50x64).
It is available at https://huggingface.co/datasets/laion/laion-coco

### Download the metadata

Download from https://huggingface.co/datasets/laion/laion-coco

```bash
mkdir -p laion-coco && cd laion-coco/

for i in {0..127}; do 
    wget "https://huggingface.co/datasets/laion/laion-coco/resolve/main/part-$(printf "%05d" $i)-2256f782-126f-4dc6-b9c6-e6757637749d-c000.snappy.parquet"
done

cd ..
```

### Download the images with img2dataset

```bash
img2dataset --url_list laion-coco --input_format "parquet"\
         --url_col "URL" --caption_col "TEXT" --output_format webdataset\
           --output_folder laion-coco-output --processes_count 16 --thread_count 64 --image_size 256\
            --resize_only_if_bigger=True --resize_mode="keep_ratio" --skip_reencode=True \
             --save_additional_columns '["similarity","hash","punsafe","pwatermark","top_caption","all_captions","all_similarities"]' --enable_wandb True
```
