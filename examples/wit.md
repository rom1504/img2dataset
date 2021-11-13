## Wikipedia-based Image Text

[Wikipedia-based Image Text](https://github.com/google-research-datasets/wit/blob/main/DATA.md) is a dataset of 12 million image and caption.

### Download the metadata

```bash
mkdir wit-meta
cd wit-meta
for i in {00000..00009}
do
    aria2c -x 16 https://storage.googleapis.com/gresearch/wit/wit_v1.train.all-$i-of-00010.tsv.gz
done
```

### Download the images with img2dataset

```
img2dataset --url_list wit-meta --input_format "tsv.gz"\
         --url_col "image_url" --caption_col "caption_alt_text_description" --output_format webdataset\
           --output_folder wit-data --processes_count 16 --thread_count 16 --image_size 256\
             --save_additional_columns '["language","page_url","page_title","section_title","hierarchical_section_title","caption_reference_description","caption_attribution_description","mime_type","is_main_image","attribution_passes_lang_id","page_changed_recently","context_page_description","context_section_description"]' --enable_wandb True
```

### Benchmark




