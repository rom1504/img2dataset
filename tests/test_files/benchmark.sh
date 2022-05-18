set -e
#rm -rf bench
time img2dataset --processes_count 16 --thread_count 128 --url_list=test_10000.parquet --image_size=256 --output_folder=bench \
--output_format="files" --input_format "parquet" --url_col "URL" --caption_col "TEXT" --enable_wandb True --number_sample_per_shard 1000 \
--distributor multiprocessing
#rm -rf bench