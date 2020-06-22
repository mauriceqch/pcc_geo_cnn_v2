import numpy as np


def to_bytes(x, dtype):
    iinfo = np.iinfo(dtype)
    x = np.array(x, dtype=dtype)
    assert np.all(x <= iinfo.max), f'Overflow {x} {iinfo}'
    assert np.all(iinfo.min <= x), f'Underflow {x} {iinfo}'
    return x.tobytes()


def scalar_to_bytes(x, dtype):
    return to_bytes([x], dtype)


def read_from_buffer(f, n, dtype):
    return np.frombuffer(f.read(int(np.dtype(dtype).itemsize * n)), dtype=dtype)


def save_compressed_file(binstr, data_b_list, resolution, octree_level):
    """Saves an octree partitioned point cloud and its partition bitstreams as an unified bitstream"""
    resolution_b = scalar_to_bytes(resolution, np.uint16)
    level_b = scalar_to_bytes(octree_level, np.uint8)
    n_blocks_b = scalar_to_bytes(len(data_b_list), np.uint16)
    n_strings_b = scalar_to_bytes(len(data_b_list[0][0]), np.uint8)
    n_binstr_b = scalar_to_bytes(len(binstr), np.uint16)
    binstr_b = to_bytes(binstr, np.uint8)
    ret = resolution_b + level_b + n_blocks_b + n_strings_b + n_binstr_b + binstr_b
    for strings, best_threshold_idx in data_b_list:
        best_threshold_idx_b = scalar_to_bytes(best_threshold_idx, np.uint8)
        ret += best_threshold_idx_b
        for s in strings:
            n_bytes_b = scalar_to_bytes(len(s), np.uint16)
            ret += n_bytes_b + s
    return ret


def load_compressed_file(f):
    """Loads an octree partitioned point cloud unified bitstream"""
    blocks = []
    resolution = read_from_buffer(f, 1, np.uint16)[0]
    level = read_from_buffer(f, 1, np.uint8)[0]
    n_blocks = read_from_buffer(f, 1, np.uint16)[0]
    n_strings = read_from_buffer(f, 1, np.uint8)[0]
    n_binstr = read_from_buffer(f, 1, np.uint16)[0]
    binstr = read_from_buffer(f, n_binstr, np.uint8)
    for _ in range(n_blocks):
        best_threshold_idx = read_from_buffer(f, 1, np.uint8)[0]
        strings = []
        for i in range(n_strings):
            n_bytes = read_from_buffer(f, 1, np.uint16)[0]
            string = f.read(int(n_bytes))
            strings.append(string)
        blocks.append((strings, best_threshold_idx))
    file_end = f.read()
    assert file_end == b'', f'File not read completely file_end {file_end}'

    return resolution, level, binstr, blocks