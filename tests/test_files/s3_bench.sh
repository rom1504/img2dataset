## this benchmarks uses parquet files from https://github.com/rom1504/cah-prepro

#rm -rf /media/nvme/sample_unresized_laion400m
aws s3 rm --recursive s3://laion-watermark/my_img2dataset_test/
img2dataset --url_list "/media/hd/testing/cah_400M_meta" --input_format "parquet"\
 --url_col "URL" --caption_col "TEXT" --output_format webdataset\
  --output_folder "s3://laion-watermark/my_img2dataset_test" --processes_count 16 --thread_count 64 --image_size 256\
  --save_additional_columns '["NSFW","similarity","LICENSE"]' --enable_wandb True --distributor pyspark
