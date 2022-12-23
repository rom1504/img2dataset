## this benchmarks uses parquet files from https://github.com/rom1504/cah-prepro

rm -rf /media/hd/testing/tmp_test
python -m img2dataset.batch.preparer --url_list /media/hd/testing/cah_400M_meta --input_format "parquet"\
 --url_col "URL" --caption_col "TEXT" --number_sample_per_shard 100 \
  --output_folder /media/hd/testing/tmp_test \
  --save_additional_columns '["NSFW","similarity","LICENSE"]'