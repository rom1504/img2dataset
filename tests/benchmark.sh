set -e
time img2dataset --url_list=test_10000.parquet --image_size=256 --output_folder=bench --thread_count=256 \
--output_format="webdataset" --input_format "parquet" --url_col "URL" --caption_col "TEXT"
#rm -rf bench