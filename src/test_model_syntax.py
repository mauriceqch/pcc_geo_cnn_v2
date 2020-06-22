from io import BytesIO

import pytest
from numpy.testing import assert_array_equal
from model_syntax import save_compressed_file, load_compressed_file


def test_save_load():
    binstr = [1, 2, 3]
    data_b_list = [[[b'abc', b'efg'], 35], [[b'xyz', b'uvw'], 7]]
    octree_level = 4
    resolution = 512
    c = save_compressed_file(binstr, data_b_list, resolution, octree_level)
    resolution_dec, level_dec, binstr_dec, blocks_dec = load_compressed_file(BytesIO(c))
    assert_array_equal(binstr, binstr_dec)
    for data, data_dec in zip(data_b_list, blocks_dec):
        assert_array_equal(data[0], data_dec[0])
        assert data[1] == data_dec[1]
    assert octree_level == level_dec
    assert resolution == resolution_dec

    data_b_list_overflow = [[[b'abc'] * (2 ** 16 + 1), 35]]
    c = save_compressed_file(binstr, data_b_list_overflow, resolution, octree_level)
    with pytest.raises(AssertionError):
        load_compressed_file(BytesIO(c))

    octree_level_underflow = -1
    c = save_compressed_file(binstr, data_b_list_overflow, resolution, octree_level_underflow)
    with pytest.raises(AssertionError):
        load_compressed_file(BytesIO(c))
