
rm -rf /media/hd/testing/tmp_test
img2dataset --url_list /media/hd2/subcaptions/sbu-captions-all.json --input_format "json" --url_col "image_urls"\
 --caption_col "captions" --output_format webdataset --downloader async --encode_format webp\
 --output_folder /media/hd/testing/tmp_test --processes_count 16 --thread_count 128 --image_size 256 --enable_wandb True