from img2dataset.reader import Reader
import os
from fixtures import generate_input_file, setup_fixtures
import pytest
import math
import time
import gc
import psutil


def current_memory_usage():
    return psutil.Process().memory_info().rss / 1024 / 1024


@pytest.mark.parametrize(
    "input_format", ["txt", "csv", "tsv", "tsv.gz", "json", "parquet",],
)
def test_reader(input_format):
    """Tests whether Reader class works as expected."""
    expected_count = 10 ** 5 + 5312
    test_folder, test_list, _ = setup_fixtures(count=expected_count)
    prefix = input_format + "_"
    url_list_name = os.path.join(test_folder, prefix + "url_list")
    url_list_name = generate_input_file(input_format, url_list_name, test_list)

    start_shard_id = 37
    batch_size = 10000
    reader = Reader(
        url_list=url_list_name,
        input_format=input_format,
        url_col="url",
        caption_col=None if input_format == "txt" else "caption",
        save_additional_columns=None,
        number_sample_per_shard=batch_size,
        start_shard_id=start_shard_id,
    )

    if input_format == "txt":
        assert reader.column_list == ["url"]
    else:
        assert reader.column_list == ["caption", "url"]
    last_shard_num = math.ceil(expected_count / batch_size) - 1

    total_sample_count = 0
    start_time = time.time()
    initial_memory_usage = current_memory_usage()
    for incremental_shard_id, (shard_id, shard) in enumerate(reader):
        assert incremental_shard_id + start_shard_id == shard_id
        total_sample_count += len(shard)
        if last_shard_num == incremental_shard_id:
            assert len(shard) <= batch_size
        else:
            assert len(shard) == batch_size

        begin_expected = incremental_shard_id * batch_size
        end_expected = (incremental_shard_id + 1) * batch_size

        expected_shard = list(enumerate(test_list[begin_expected:end_expected]))
        if input_format == "txt":
            expected_shard = [(i, (url,)) for i, (_, url) in expected_shard]
        assert shard == expected_shard
        current_usage = current_memory_usage()
        assert current_usage - initial_memory_usage < 100
        del expected_shard
        del shard

    del reader

    assert total_sample_count == expected_count

    total_time = time.time() - start_time
    print("Total time:", total_time)
    assert total_time <= 1.0

    gc.collect()

    final_memory_usage = current_memory_usage()
    assert final_memory_usage - initial_memory_usage < 100

    os.remove(url_list_name)
