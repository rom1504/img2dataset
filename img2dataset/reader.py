"""Reader is module to read the url list and return shards"""

import glob
import os
import pandas as pd
import tempfile
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

    def _save_to_arrow(self, input_file, tmp_dir):
        """Read the input file and save to arrow files in a temporary directory"""
        if self.input_format == "txt":
            with open(input_file, encoding="utf-8") as file:
                df = pd.DataFrame([(url.rstrip(),) for url in file.readlines()], columns=self.column_list)
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
        else:
            assert False, f"Unexpected input format ({self.input_format})."

        df = df.rename(columns={self.caption_col: "caption", self.url_col: "url"})
        df = df.where(pd.notnull(df), None)

        number_samples = len(df)

        number_shards = math.ceil(len(df) / self.number_sample_per_shard)

        shards = []
        for shard_id in range(number_shards):
            begin_shard = shard_id * self.number_sample_per_shard
            end_shard = min(number_samples, (1 + shard_id) * self.number_sample_per_shard)
            df_shard = df[begin_shard:end_shard][self.column_list]
            df_shard = df_shard.reset_index(drop=True)
            file_name = tmp_dir + f"/{shard_id + self.start_shard_id}.feather"
            df_shard.to_feather(file_name)
            shards.append((shard_id, file_name))
        del df

        return shards

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

            with tempfile.TemporaryDirectory("img2dataset") as tmp_dir:
                shards = self._save_to_arrow(input_file, tmp_dir)
                num_shard = 0
                for num_shard, arrow_file in shards:
                    df = pd.read_feather(arrow_file)
                    images_to_dl = df[self.column_list].to_records(index=False).tolist()
                    del df

                    yield (
                        num_shard + self.start_shard_id,
                        list(enumerate(images_to_dl)),
                    )

                    num_shard += 1
                self.start_shard_id += num_shard
                del images_to_dl
