## this benchmarks uses parquet files from https://github.com/rom1504/cah-prepro

#rm -rf /media/hd/testing/tmp_test
python -m img2dataset.worker  --arrow_file /media/hd/testing/tmp_test/_tmp/205.feather --shard_id 205 --input_format parquet \
 --caption_col "TEXT"  --output_format webdataset --output_folder /media/hd/testing/tmp_test --thread_count 64 --image_size 256\
  --save_additional_columns '["NSFW","similarity","LICENSE"]' 