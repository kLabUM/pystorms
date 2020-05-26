import pystorms
import pytest


def test_load_network():
    network = pystorms.networks.load_network("gamma")
    assert "inp" == network[-3:]
    network = pystorms.networks.load_network("epsilon")
    assert "inp" == network[-3:]
    network = pystorms.networks.load_network("theta")
    assert "inp" == network[-3:]
    network = pystorms.networks.load_network("delta")
    assert "inp" == network[-3:]
    with pytest.raises(ValueError):
        pystorms.networks.load_network("purushotham")
