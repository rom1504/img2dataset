"""Reader is module to read the url list and return shards"""

from multiprocessing.pool import ThreadPool
import math
import fsspec
import time
import pyarrow.parquet as pq
import pyarrow.csv as csv_pa
import pyarrow.json as json_pa
import pyarrow as pa
import pandas as pd


class Reader:
    """
    The reader class reads an url list and returns shards
    It provides an iter method
    It provides attributes:
    - column_list: the list of columns to read
    - input_format: the format of the input file
    - url_col: the column name of the url
    - caption_col: the column name of the caption
    - verify_hash_col: the column containing the hash to verify.
    - verify_hash_type: the type of hash to verify.
    - save_additional_columns: the list of additional columns to save
    - number_sample_per_shard: the number of samples per shard
    - done_shards: a set of already done shards
    - start_shard_id: the shard id to begin downloading from
    """

    def __init__(
        self,
        url_list,
        input_format,
        url_col,
        caption_col,
        verify_hash_col,
        verify_hash_type,
        save_additional_columns,
        number_sample_per_shard,
        done_shards,
        tmp_path,
        start_shard_id: int = 0,
    ) -> None:
        self.input_format = input_format
        self.url_col = url_col
        self.caption_col = caption_col
        self.verify_hash_col = verify_hash_col
        self.verify_hash_type = verify_hash_type
        self.save_additional_columns = save_additional_columns
        self.number_sample_per_shard = number_sample_per_shard
        self.done_shards = done_shards
        self.start_shard_id = start_shard_id

        fs, url_path = fsspec.core.url_to_fs(url_list)
        self.fs = fs
        self.tmp_path = tmp_path

        if fs.isdir(url_path):
            self.input_files = sorted(fs.glob(url_path.rstrip("/") + "/*." + input_format))
            if len(self.input_files) == 0:
                raise ValueError(f"No file found at path {url_path} with extension {input_format}")
        else:
            self.input_files = [url_path]

        if self.input_format in ["txt", "txt.gz"]:
            self.column_list = ["url"]
        elif self.input_format in ["json", "json.gz", "jsonl", "jsonl.gz", "csv", "csv.gz", "tsv", "tsv.gz", "parquet"]:
            self.column_list = self.save_additional_columns if self.save_additional_columns is not None else []
            if self.caption_col is not None:
                self.column_list = self.column_list + ["caption"]
            if self.verify_hash_col is not None:
                if self.verify_hash_type in ["md5", "sha256", "sha512"]:
                    self.column_list = self.column_list + [self.verify_hash_type]
                else:
                    raise ValueError(f"Invalid hash type {self.verify_hash_type}")
            self.column_list = self.column_list + ["url"]
        else:
            raise ValueError(f"Invalid input format {self.input_format}")

    def _save_to_arrow(self, input_file, start_shard_id):
        """Read the input file and save to arrow files in a temporary directory"""
        if self.input_format in [
            "txt",
            "txt.gz",
            "csv",
            "csv.gz",
            "tsv",
            "tsv.gz",
            "json",
            "json.gz",
            "jsonl",
            "jsonl.gz",
        ]:
            compression = None
            if self.input_format.endswith(".gz"):
                compression = "gzip"
            with self.fs.open(input_file, encoding="utf-8", mode="rb", compression=compression) as file:
                if self.input_format in ["txt", "txt.gz"]:
                    df = csv_pa.read_csv(file, read_options=csv_pa.ReadOptions(column_names=["url"]))
                elif self.input_format in ["json", "json.gz"]:
                    df = pa.Table.from_pandas(pd.read_json(file))
                elif self.input_format in ["csv", "csv.gz"]:
                    df = csv_pa.read_csv(file)
                elif self.input_format in ["tsv", "tsv.gz"]:
                    df = csv_pa.read_csv(file, parse_options=csv_pa.ParseOptions(delimiter="\t"))
                elif self.input_format in ["jsonl", "jsonl.gz"]:
                    df = json_pa.read_json(file)
                else:
                    raise ValueError(f"Unknown input format {self.input_format}")
        elif self.input_format == "parquet":
            with self.fs.open(input_file, mode="rb") as file:
                columns_to_read = [self.url_col]
                if self.caption_col is not None:
                    columns_to_read += [self.caption_col]
                if self.verify_hash_col is not None:
                    columns_to_read += [self.verify_hash_col]
                if self.save_additional_columns is not None:
                    columns_to_read += self.save_additional_columns
                df = pq.read_table(file, columns=columns_to_read)
        else:
            raise ValueError(f"Unknown input format {self.input_format}")

        column_names = df.column_names
        if self.caption_col is not None:
            column_names = [c if c != self.caption_col else "caption" for c in column_names]

        if self.verify_hash_col is not None:
            column_names = [c if c != self.verify_hash_col else self.verify_hash_type for c in column_names]

        column_names = [c if c != self.url_col else "url" for c in column_names]

        df = df.rename_columns(column_names)

        number_samples = df.num_rows

        number_shards = math.ceil(df.num_rows / self.number_sample_per_shard)
        shards_to_write = [
            (start_shard_id + shard_id, shard_id)
            for shard_id in range(number_shards)
            if start_shard_id + shard_id not in self.done_shards
        ]
        if len(shards_to_write) == 0:
            return [], number_shards

        def write_shard(t):
            full_shard_id, shard_id = t
            begin_shard = shard_id * self.number_sample_per_shard
            end_shard = min(number_samples, (1 + shard_id) * self.number_sample_per_shard)
            df_shard = df.slice(begin_shard, end_shard - begin_shard).select(self.column_list)
            tmp_file = self.tmp_path + f"/{full_shard_id}.feather"
            for i in range(10):
                try:
                    fs, tmp_path = fsspec.core.url_to_fs(tmp_file)
                    with fs.open(tmp_path, "wb") as file:
                        with pa.ipc.new_file(file, df_shard.schema) as writer:
                            writer.write_table(df_shard)
                    return (full_shard_id, tmp_file)
                except Exception as e:  # pylint: disable=broad-except
                    if i != 9:
                        print("retrying to write to file due to error:", e)
                        time.sleep(1)
                    else:
                        raise e
            # can't reach here
            raise ValueError("Failed to write to file.")

        for i in range(10):
            shards = []
            # thread pool to make it faster to write files to low latency file systems (ie s3, hdfs)
            try:
                with ThreadPool(32) as thread_pool:
                    for shard in thread_pool.imap_unordered(write_shard, shards_to_write):
                        shards.append(shard)
                break
            except Exception as e:  # pylint: disable=broad-except
                if i != 9:
                    print("retrying whole sharding to write to files due to error:", e)
                    time.sleep(2 * i)
                else:
                    raise e

        shards.sort(key=lambda k: k[0])

        del df

        return shards, number_shards

    def __iter__(self):
        """
        Iterate over shards, yield shards of size number_sample_per_shard or less for the last one
        Each shard is a tuple (shard_id, shard)
        shard is a tuple (sample id, sample)
        sample is a tuple of the columns
        """
        start_shard_id = self.start_shard_id
        for i, input_file in enumerate(self.input_files):
            print("Sharding file number " + str(i + 1) + " of " + str(len(self.input_files)) + " called " + input_file)

            shards, number_shards = self._save_to_arrow(input_file, start_shard_id)
            print("File sharded in " + str(len(shards)) + " shards")
            print(
                "Downloading starting now, check your bandwidth speed (with bwm-ng)"
                "your cpu (with htop), and your disk usage (with iotop)!"
            )

            for shard_id, arrow_file in shards:
                yield (
                    shard_id,
                    arrow_file,
                )
            start_shard_id += number_shards
