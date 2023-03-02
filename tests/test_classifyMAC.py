import pytest
from rulesBased.classifyMAC import classifyMAC

def test_classifications():
    assert classifyMAC(bytes.fromhex('000000000000')) == 3
    assert classifyMAC(bytes.fromhex('FFFFFFFFFFFF')) == 3
    assert classifyMAC(bytes.fromhex('B827EB000000')) == 2      # raspberry pi
    assert classifyMAC(bytes.fromhex('608B0E000000')) == 1      # MAC address in list but not 'good'
    assert classifyMAC(bytes.fromhex('CA7CA7000000')) == 0      # MAC address not in list

def test_valueErrors():
    with pytest.raises(ValueError):
        classifyMAC(bytes.fromhex('00'))                        # too short of an address
        classifyMAC(bytes.fromhex('11223344556677'))            # too long of an address

