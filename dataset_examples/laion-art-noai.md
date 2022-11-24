## Laion-art

Laion art is a 8M samples laion5B subset with aesthetic > 8 pwatermark < 0.8 punsafe < 0.5
See [full description](https://github.com/LAION-AI/laion-datasets/blob/main/laion-aesthetic.md)

It is available at https://huggingface.co/datasets/laion/laion-art

A good use case is to train an image generation model. However, concerns have been raised about how to ethically source training data at scale for such purposes, especially where creator consent may not have been expressed in advance of dataset construction. One solution is to allow artists to opt their images out of such usage by sending HTTP header directives when image content is downloaded. img2dataset can be configured to automatically respect such directives.

### Download the metadata

Download from [https://huggingface.co/datasets/laion/laion1B-nolang-aesthetic 
https://huggingface.co/datasets/laion/laion2B-en-aesthetic
https://huggingface.co/datasets/laion/laion2B-multi-aesthetic](https://huggingface.co/datasets/laion/laion-art)

```
wget https://huggingface.co/datasets/laion/laion-art/resolve/main/laion-art.parquet
```

### Download the images with img2dataset, respecting noai and noindex directives

```
img2dataset --url_list laion-art --input_format "parquet"\
         --url_col "URL" --caption_col "TEXT" --output_format webdataset\
           --output_folder laion-high-resolution --processes_count 16 --thread_count 64 --image_size 384\
            --resize_only_if_bigger=True --resize_mode="keep_ratio" --skip_reencode=True \
             --save_additional_columns '["similarity","hash","punsafe","pwatermark","aesthetic","LANGUAGE"]' --enable_wandb True \
              --user_agent_token img2dataset --disallowed_header_directives '["noai", "noindex"]'
```

### Benchmark
