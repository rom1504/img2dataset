rm -rf /media/hd/testing/tmp_test

python -m img2dataset.service.dataloader \
   --output_format webdataset --processes_count 16 --thread_count 64 --image_size 256\
  --save_additional_columns '["NSFW","similarity","LICENSE"]' \
  -url_list /media/hd/testing/cah_400M_meta --input_format "parquet"\
 --url_col "URL" --caption_col "TEXT" --number_sample_per_shard 1000 \
  --output_folder /media/hd/testing/tmp_test
