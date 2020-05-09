import pystorms
import pytest


def test_load_binary():
    with pytest.raises(ValueError):
        pystorms.binaries.load_binary("purushotham")
