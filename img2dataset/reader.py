"""Reader is module to read the url list and return shards"""

import glob
import os
import pandas as pd
import math


class Reader:
    """
    The reader class reads an url list and returns shards
    It provides an iter method
    It provides attributes:
    - column_list: the list of columns to read
    - input_format: the format of the input file
    - url_col: the column name of the url
    - caption_col: the column name of the caption
    - save_additional_columns: the list of additional columns to save
    - number_sample_per_shard: the number of samples per shard
    - start_shard_id: the id of the first shard
    """

    def __init__(
        self,
        url_list,
        input_format,
        url_col,
        caption_col,
        save_additional_columns,
        number_sample_per_shard,
        start_shard_id,
    ) -> None:
        self.input_format = input_format
        self.url_col = url_col
        self.caption_col = caption_col
        self.save_additional_columns = save_additional_columns
        self.number_sample_per_shard = number_sample_per_shard
        self.start_shard_id = start_shard_id

        if os.path.isdir(url_list):
            self.input_files = sorted(glob.iglob(url_list + "/*." + input_format))
        else:
            self.input_files = [url_list]

        if self.input_format == "txt":
            self.column_list = ["url"]
        elif self.input_format in ["json", "csv", "tsv", "tsv.gz", "parquet"]:
            self.column_list = self.save_additional_columns if self.save_additional_columns is not None else []
            if self.caption_col is not None:
                self.column_list = self.column_list + ["caption", "url"]
            else:
                self.column_list = self.column_list + ["url"]

    def __iter__(self):
        """
        Iterate over shards, yield shards of size number_sample_per_shard or less for the last one
        Each shard is a tuple (shard_id, shard)
        shard is a tuple (sample id, sample)
        sample is a tuple of the columns
        """
        for i, input_file in enumerate(self.input_files):
            print(
                "Downloading file number " + str(i + 1) + " of " + str(len(self.input_files)) + " called " + input_file
            )
            print("Loading the input file")

            if self.input_format == "txt":
                images_to_dl = []
                with open(input_file, encoding="utf-8") as file:
                    images_to_dl = [(url.rstrip(),) for url in file.readlines()]
            elif self.input_format in ["json", "csv", "tsv", "tsv.gz", "parquet"]:
                if self.input_format == "json":
                    df = pd.read_json(input_file)
                elif self.input_format == "csv":
                    df = pd.read_csv(input_file)
                elif self.input_format in ("tsv", "tsv.gz"):
                    df = pd.read_table(input_file)
                elif self.input_format == "parquet":
                    columns_to_read = [self.url_col]
                    if self.caption_col is not None:
                        columns_to_read += [self.caption_col]
                    if self.save_additional_columns is not None:
                        columns_to_read += self.save_additional_columns
                    df = pd.read_parquet(input_file, columns=columns_to_read)
                df = df.rename(columns={self.caption_col: "caption", self.url_col: "url"})
                df = df.where(pd.notnull(df), None)
                images_to_dl = df[self.column_list].to_records(index=False).tolist()
                del df
            else:
                assert False, f"Unexpected input format ({self.input_format})."

            number_samples = len(images_to_dl)
            number_shards = math.ceil(number_samples / self.number_sample_per_shard)
            print(
                f"Splitting the {number_samples} samples in {number_shards}"
                f"shards of size {self.number_sample_per_shard}"
            )
            for shard_id in range(number_shards):
                begin_shard = shard_id * self.number_sample_per_shard
                end_shard = min(number_samples, (1 + shard_id) * self.number_sample_per_shard)
                yield (
                    shard_id + self.start_shard_id,
                    list(enumerate(images_to_dl[begin_shard:end_shard])),
                )
            self.start_shard_id += number_shards
            del images_to_dl
            print("Done sharding the input file")
