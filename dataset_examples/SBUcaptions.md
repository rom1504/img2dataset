## SBU Captions

[SBU Captions]([https://www.cs.rice.edu/~vo9/sbucaptions/sbu-captions]) is a large-scale dataset that contains 860K image-text pairs as well as many other meta-attributes to increase the usability to train various models. This dataset is one of the key benchmark datasets.

### Download the metadata


```
wget https://www.cs.rice.edu/~vo9/sbucaptions/sbu-captions-all.tar.gz
tar -xvzf sbu-captions-all.tar.gz

```

### Download the images with img2dataset

```
img2dataset --url_list sbu-captions-all.json --input_format "json" --url_col "image_urls" --caption_col "captions" --output_format webdataset --output_folder sbucaptions --processes_count 16 --thread_count 64 --image_size 256

```

### Benchmark
