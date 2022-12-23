python -m img2dataset.service.launcher --input_format parquet \
 --caption_col "TEXT"  --output_format webdataset --processes_count 2 --thread_count 64 --image_size 256\
  --save_additional_columns '["NSFW","similarity","LICENSE"]' 