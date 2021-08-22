rm -rf /media/hd/testing/tmp_test
img2dataset --url_list /media/hd/testing/part_one.parquet --input_format "parquet"\
 --url_col "URL" --caption_col "TEXT" --output_format webdataset\
  --output_folder /media/hd/testing/tmp_test --processes_count 16 --thread_count 128 --image_size 256