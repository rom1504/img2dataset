""""writer module handle writing the images to disk"""

import webdataset as wds
import json
import pyarrow.parquet as pq
import pyarrow as pa
import fsspec
import os
import numpy as np


class BufferedParquetWriter:
    """Write samples to parquet files incrementally with a buffer"""

    def __init__(self, output_file, schema, buffer_size=100):
        self.buffer_size = buffer_size
        self.schema = schema
        self._initiatlize_buffer()
        fs, output_path = fsspec.core.url_to_fs(output_file)

        self.output_fd = fs.open(output_path, "wb")
        self.parquet_writer = pq.ParquetWriter(self.output_fd, schema)

    def _initiatlize_buffer(self):
        self.current_buffer_size = 0
        self.buffer = {k: [] for k in self.schema.names}

    def _add_sample_to_buffer(self, sample):
        for k in self.schema.names:
            self.buffer[k].append(sample[k])
        self.current_buffer_size += 1

    def write(self, sample):
        if self.current_buffer_size >= self.buffer_size:
            self.flush()
        self._add_sample_to_buffer(sample)

    def flush(self):
        """Write the buffer to disk"""
        if self.current_buffer_size == 0:
            return

        df = pa.Table.from_pydict(self.buffer, self.schema)
        self.parquet_writer.write_table(df)
        self._initiatlize_buffer()

    def close(self):
        self.flush()
        if self.parquet_writer is not None:
            self.parquet_writer.close()
            self.parquet_writer = None
            self.output_fd.close()


class ParquetSampleWriter:
    """ParquetSampleWriter is a image+caption writer to parquet"""

    def __init__(self, shard_id, output_folder, save_caption, oom_shard_count, schema):
        self.oom_shard_count = oom_shard_count
        schema = schema.append(pa.field("jpg", pa.binary()))
        shard_name = "{shard_id:0{oom_shard_count}d}".format(  # pylint: disable=consider-using-f-string
            shard_id=shard_id, oom_shard_count=oom_shard_count
        )
        output_file = f"{output_folder}/{shard_name}.parquet"
        self.buffered_parquet_writer = BufferedParquetWriter(output_file, schema, 100)
        self.save_caption = save_caption

    def write(self, img_str, key, caption, meta):
        """Keep sample in memory then write to disk when close() is called"""
        if img_str is not None:
            sample = {"key": key, "jpg": img_str}
            if self.save_caption:
                sample["txt"] = str(caption) if caption is not None else ""
        else:
            sample = {"key": key, "jpg": None}
            if self.save_caption:
                sample["txt"] = None
        sample.update(meta)
        self.buffered_parquet_writer.write(sample)

    def close(self):
        self.buffered_parquet_writer.close()


class WebDatasetSampleWriter:
    """WebDatasetSampleWriter is a image+caption writer to webdataset"""

    def __init__(self, shard_id, output_folder, save_caption, oom_shard_count, schema):
        self.oom_shard_count = oom_shard_count
        shard_name = "{shard_id:0{oom_shard_count}d}".format(  # pylint: disable=consider-using-f-string
            shard_id=shard_id, oom_shard_count=oom_shard_count
        )
        self.shard_id = shard_id
        fs, output_path = fsspec.core.url_to_fs(output_folder)
        self.tar_fd = fs.open(f"{output_path}/{shard_name}.tar", "wb")
        self.tarwriter = wds.TarWriter(self.tar_fd)
        self.save_caption = save_caption
        self.buffered_parquet_writer = BufferedParquetWriter(output_folder + "/" + shard_name + ".parquet", schema, 100)

    def write(self, img_str, key, caption, meta):
        """write sample to tars"""
        if img_str is not None:
            sample = {"__key__": key, "jpg": img_str}
            if self.save_caption:
                sample["txt"] = str(caption) if caption is not None else ""
            # some meta data may not be JSON serializable
            for k, v in meta.items():
                if isinstance(v, np.ndarray):
                    meta[k] = v.tolist()
            sample["json"] = json.dumps(meta, indent=4)
            self.tarwriter.write(sample)
        self.buffered_parquet_writer.write(meta)

    def close(self):
        self.buffered_parquet_writer.close()
        self.tarwriter.close()
        self.tar_fd.close()


class TFRecordSampleWriter:
    """TFRecordSampleWriter is a image+caption writer to TFRecord"""

    def __init__(self, shard_id, output_folder, save_caption, oom_shard_count, schema):
        try:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            import tensorflow_io as _  # pylint: disable=import-outside-toplevel
            from tensorflow.python.training.training import (  # pylint: disable=import-outside-toplevel
                BytesList,
                Int64List,
                FloatList,
                Example,
                Features,
                Feature,
            )
            from tensorflow.python.lib.io.tf_record import TFRecordWriter  # pylint: disable=import-outside-toplevel

            self._BytesList = BytesList  # pylint: disable=invalid-name
            self._Int64List = Int64List  # pylint: disable=invalid-name
            self._FloatList = FloatList  # pylint: disable=invalid-name
            self._Example = Example  # pylint: disable=invalid-name
            self._Features = Features  # pylint: disable=invalid-name
            self._Feature = Feature  # pylint: disable=invalid-name
        except ImportError as e:
            raise ModuleNotFoundError(
                "tfrecords require tensorflow and tensorflow_io to be installed."
                "Run `pip install tensorflow tensorflow_io`."
            ) from e

        self.oom_shard_count = oom_shard_count
        shard_name = "{shard_id:0{oom_shard_count}d}".format(  # pylint: disable=consider-using-f-string
            shard_id=shard_id, oom_shard_count=oom_shard_count
        )
        self.shard_id = shard_id
        self.tf_writer = TFRecordWriter(f"{output_folder}/{shard_name}.tfrecord")
        self.save_caption = save_caption
        self.buffered_parquet_writer = BufferedParquetWriter(output_folder + "/" + shard_name + ".parquet", schema, 100)

    def write(self, img_str, key, caption, meta):
        """Write a sample using tfrecord writer"""
        if img_str is not None:
            sample = {
                "key": self._bytes_feature(key.encode()),
                "jpg": self._bytes_feature(img_str),
            }
            if self.save_caption:
                sample["txt"] = self._bytes_feature(str(caption) if caption is not None else "")
            for k, v in meta.items():
                sample[k] = self._feature(v)
            tf_example = self._Example(features=self._Features(feature=sample))
            self.tf_writer.write(tf_example.SerializeToString())
        self.buffered_parquet_writer.write(meta)

    def close(self):
        self.buffered_parquet_writer.close()
        self.tf_writer.close()

    def _feature(self, value):
        """Convert to proper feature type"""
        if isinstance(value, int):
            return self._int64_feature(value)
        elif isinstance(value, float):
            return self._float_feature(value)
        else:
            return self._bytes_feature(value)

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if value is None:
            value = ""
        if isinstance(value, str):
            value = value.encode()
        return self._Feature(bytes_list=self._BytesList(value=[value]))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return self._Feature(float_list=self._FloatList(value=[value]))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return self._Feature(int64_list=self._Int64List(value=[value]))


class FilesSampleWriter:
    """FilesSampleWriter is a caption+image writer to files"""

    def __init__(self, shard_id, output_folder, save_caption, oom_shard_count, schema):
        self.oom_shard_count = oom_shard_count
        shard_name = "{shard_id:0{oom_shard_count}d}".format(  # pylint: disable=consider-using-f-string
            shard_id=shard_id, oom_shard_count=oom_shard_count
        )
        self.shard_id = shard_id
        self.fs, self.subfolder = fsspec.core.url_to_fs(f"{output_folder}/{shard_name}")
        if not self.fs.exists(self.subfolder):
            self.fs.mkdir(self.subfolder)
        self.save_caption = save_caption
        self.buffered_parquet_writer = BufferedParquetWriter(output_folder + "/" + shard_name + ".parquet", schema, 100)

    def write(self, img_str, key, caption, meta):
        """Write sample to disk"""
        if img_str is not None:
            filename = f"{self.subfolder}/{key}.jpg"
            with self.fs.open(filename, "wb") as f:
                f.write(img_str)
            if self.save_caption:
                caption = str(caption) if caption is not None else ""
                caption_filename = f"{self.subfolder}/{key}.txt"
                with self.fs.open(caption_filename, "w") as f:
                    f.write(str(caption))

            # some meta data may not be JSON serializable
            for k, v in meta.items():
                if isinstance(v, np.ndarray):
                    meta[k] = v.tolist()
            j = json.dumps(meta, indent=4)
            meta_filename = f"{self.subfolder}/{key}.json"
            with self.fs.open(meta_filename, "w") as f:
                f.write(j)
        self.buffered_parquet_writer.write(meta)

    def close(self):
        self.buffered_parquet_writer.close()


class DummySampleWriter:
    """Does not write"""

    def __init__(self, shard_id, output_folder, save_caption, oom_shard_count, schema):
        pass

    def write(self, img_str, key, caption, meta):
        pass

    def close(self):
        pass
