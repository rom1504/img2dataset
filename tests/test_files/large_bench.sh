## this benchmarks uses parquet files from https://github.com/rom1504/cah-prepro

rm -rf /media/hd/testing/tmp_test

img2dataset --url_list /media/hd2/laion-a/laion-a-2-77000000.parquet   --output_folder /media/hd/testing/tmp_test \
  --processes_count=16 \
    --thread_count=16 \
    --retries=0 \
    --encode_quality=100 \
    --resize_mode=no \
    --skip_reencode=True \
    --output_format="webdataset" \
    --input_format="parquet" \
    --url_col="url" \
    --caption_col="caption" \
    --enable_wandb=True \
    --number_sample_per_shard=1000 \
    --distributor="pyspark" \
    --save_additional_columns='["similarity","punsafe","pwatermark","AESTHETIC_SCORE","width","height"]'
