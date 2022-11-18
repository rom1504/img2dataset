## COYO-700M

[COYO-700M](https://github.com/kakaobrain/coyo-dataset) is a large-scale dataset that contains 747M image-text pairs as well as many other meta-attributes to increase the usability to train various models. Our dataset follows a similar strategy to previous vision-and-language datasets, collecting many informative pairs of alt-text and its associated image in HTML documents. We expect COYO to be used to train popular large-scale foundation models complementary to other similar datasets.

### Download the metadata

Download from https://huggingface.co/datasets/kakaobrain/coyo-700m  
We are providing a [download guide](https://github.com/kakaobrain/coyo-dataset/tree/main/download)

```
mkdir coyo-700m && cd coyo-700m
for i in {00000..00127}; do wget https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-$i-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet; done
cd ..
```

### Download the images with img2dataset

```
img2dataset --url_list coyo-700m --input_format "parquet"\
         --url_col "url" --caption_col "text" --output_format webdataset\
           --output_folder coyo-700m-webdataset --processes_count 16 --thread_count 64 --image_size 384\
            --resize_only_if_bigger=True --resize_mode="keep_ratio" --skip_reencode=True \
             --save_additional_columns '["clip_similarity_vitb32","clip_similarity_vitl14","nsfw_score_opennsfw2","nsfw_score_gantman","watermark_score","aesthetic_score_laion_v2"]' --enable_wandb False
```

### Benchmark

