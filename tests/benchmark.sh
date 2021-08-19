set -e
time img2dataset --url_list=test10000.txt --image_size=256 --output_folder=bench --thread_count=200 --output_format="webdataset"
rm -rf bench