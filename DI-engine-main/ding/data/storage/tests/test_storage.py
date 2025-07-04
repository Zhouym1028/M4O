import tempfile
import pytest
import os
from os import path
from ding.data.storage import FileStorage


@pytest.mark.unittest
def test_file_storage():
    path_ = path.join(tempfile.gettempdir(), "test_storage.txt")
    try:
        storage = FileStorage(path=path_)
        storage.save("plot")
        content = storage.load()
        assert content == "plot"
    finally:
        if path.exists(path_):
            os.remove(path_)
