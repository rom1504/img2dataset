import sys
import time
from collections import Counter

import ray
from img2dataset import download

import argparse




@ray.remote
def main(args):
    download(
	processes_count=1, 
	thread_count=32,
	retries=0,
	timeout=10,
	url_list=args.url_list,
	image_size=512,
	resize_only_if_bigger=True,
	resize_mode="keep_ratio_largest",
	skip_reencode=True,
	output_folder=args.out_folder,
	output_format="webdataset",
	input_format="parquet",
	url_col="url",
	caption_col="alt",
	enable_wandb=True,
	subjob_size=48*120*2,
	number_sample_per_shard=10000,
	distributor="ray",
	oom_shard_count=8,
    compute_hash="sha256",
	save_additional_columns=["uid"]
    )

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--url_list")
	parser.add_argument("--out_folder")
	args = parser.parse_args()
	ray.init(address="localhost:6379")
	main(args)




