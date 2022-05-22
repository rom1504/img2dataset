## Laion-aesthetic

Laion aesthetic is a laion5B subset with aesthetic > 7 pwatermark < 0.8 punsafe < 0.5
See [full description](https://github.com/LAION-AI/laion-datasets/blob/main/laion-aesthetic.md)

It is available at https://huggingface.co/datasets/laion/laion1B-nolang-aesthetic 
https://huggingface.co/datasets/laion/laion2B-en-aesthetic
https://huggingface.co/datasets/laion/laion2B-multi-aesthetic

It has 52M + 51M + 17M samples

A good use case is to train an image generation model.

### Download the metadata

Download from https://huggingface.co/datasets/laion/laion1B-nolang-aesthetic 
https://huggingface.co/datasets/laion/laion2B-en-aesthetic
https://huggingface.co/datasets/laion/laion2B-multi-aesthetic

```
mkdir laion2B-en-aesthetic && cd laion2B-en-aesthetic
for i in {00000..00127}; do wget https://huggingface.co/datasets/laion/laion2B-en-aesthetic/resolve/main/part-$i-9230b837-b1e0-4254-8b88-ed2976e9cee9-c000.snappy.parquet; done
cd ..
```

Very similar for laion2B-multi and laion1B-nolang

### Download the images with img2dataset

```
img2dataset --url_list laion2B-en-aesthetic --input_format "parquet"\
         --url_col "URL" --caption_col "TEXT" --output_format webdataset\
           --output_folder laion-high-resolution --processes_count 16 --thread_count 64 --image_size 256\
            --resize_only_if_bigger=True --resize_mode="keep_ratio" --skip_reencode=True \
             --save_additional_columns '["similarity","hash","punsafe","pwatermark","aesthetic"]' --enable_wandb True
```

### Benchmark

