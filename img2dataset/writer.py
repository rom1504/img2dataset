""""writer module handle writing the images to disk"""

import webdataset as wds
import json
import os


class WebDatasetSampleWriter:
    """WebDatasetSampleWriter is a image+caption writer to webdataset"""

    def __init__(self, shard_id, output_folder, save_caption, save_metadata, oom_shard_count):
        self.oom_shard_count = oom_shard_count
        shard_name = "{shard_id:0{oom_shard_count}d}".format(shard_id=shard_id, oom_shard_count=oom_shard_count)
        self.shard_id = shard_id
        self.tarwriter = wds.TarWriter(f"{output_folder}/{shard_name}.tar")
        self.save_caption = save_caption
        self.save_metadata = save_metadata

    def write(self, img_str, key, caption, meta):
        sample = {"__key__": key, "jpg": img_str}
        if self.save_caption:
            sample["txt"] = str(caption) if caption is not None else ""
        if self.save_metadata:
            sample["json"] = json.dumps(meta, indent=4)
        self.tarwriter.write(sample)

    def close(self):
        self.tarwriter.close()


class FilesSampleWriter:
    """FilesSampleWriter is a caption+image writer to files"""

    def __init__(self, shard_id, output_folder, save_caption, save_metadata, oom_shard_count):
        self.oom_shard_count = oom_shard_count
        shard_name = "{shard_id:0{oom_shard_count}d}".format(shard_id=shard_id, oom_shard_count=oom_shard_count)
        self.shard_id = shard_id
        self.subfolder = f"{output_folder}/{shard_name}"
        if not os.path.exists(self.subfolder):
            os.mkdir(self.subfolder)
        self.save_caption = save_caption
        self.save_metadata = save_metadata

    def write(self, img_str, key, caption, meta):
        """Write sample to disk"""
        filename = f"{self.subfolder}/{key}.jpg"
        with open(filename, "wb") as f:
            f.write(img_str)
        if self.save_caption:
            caption = str(caption) if caption is not None else ""
            caption_filename = f"{self.subfolder}/{key}.txt"
            with open(caption_filename, "w") as f:
                f.write(str(caption))
        if self.save_metadata:
            j = json.dumps(meta, indent=4)
            meta_filename = f"{self.subfolder}/{key}.json"
            with open(meta_filename, "w") as f:
                f.write(j)

    def close(self):
        pass
