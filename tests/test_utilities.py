import pystorms
import pytest


def test_load_network():
    network = pystorms.utilities.load_network("gamma")
    assert "inp" == network[-3:]
    network = pystorms.utilities.load_network("epsilon")
    assert "inp" == network[-3:]
    network = pystorms.utilities.load_network("alpha")
    assert "inp" == network[-3:]
    network = pystorms.utilities.load_network("beta")
    assert "inp" == network[-3:]
    network = pystorms.utilities.load_network("theta")
    assert "inp" == network[-3:]
    network = pystorms.utilities.load_network("delta")
    assert "inp" == network[-3:]
    with pytest.raises(ValueError):
        pystorms.utilities.load_network("purushotham")


def test_load_binary():
    with pytest.raises(ValueError):
        pystorms.utilities.load_binary("purushotham")
