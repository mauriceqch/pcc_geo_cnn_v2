import re


def parse_bin_log(path):
    with open(path, 'r') as f:
        s = f.read()
    res = re.search(r'positions bitstream size ([\d]+) B \(([\d\.]+) bpp\)', s, re.MULTILINE)
    res_color = re.search(r'colors bitstream size ([\d]+) B \(([\d\.]+) bpp\)', s, re.MULTILINE)
    pos_bitstream_size_in_bytes = int(res.group(1))
    pos_bits_per_output_point = float(res.group(2))
    color_bitstream_size_in_bytes = int(res_color.group(1))
    color_bits_per_output_point = float(res_color.group(2))
    uncompressed_data_path = re.search(r'uncompressedDataPath  : "(.*)"', s, re.MULTILINE).group(1)

    return {
        'pos_bitstream_size_in_bytes': pos_bitstream_size_in_bytes,
        'pos_bits_per_output_point': pos_bits_per_output_point,
        'color_bitstream_size_in_bytes': color_bitstream_size_in_bytes,
        'color_bits_per_output_point': color_bits_per_output_point,
        'uncompressed_data_path': uncompressed_data_path,
    }


def parse_decoded_log(path):
    with open(path, 'r') as f:
        s = f.read()
    pos_bitstream_size_in_bytes = int(re.search(r'positions bitstream.*?([\d\.]+)', s, re.MULTILINE).group(1))
    color_bitstream_size_in_bytes = int(re.search(r'colors bitstream.*?([\d\.]+)', s, re.MULTILINE).group(1))
    uncompressed_data_path = re.search(r'uncompressedDataPath  : "(.*)"', s, re.MULTILINE).group(1)

    return {
        'pos_bitstream_size_in_bytes': pos_bitstream_size_in_bytes,
        'color_bitstream_size_in_bytes': color_bitstream_size_in_bytes,
        'uncompressed_data_path': uncompressed_data_path
    }


def parse_pcerror(path):
    with open(path, 'r') as f:
        s = f.read()
    try:
        d1_mae = float(re.search(r'maeF      \(p2point\): (.+)', s, re.MULTILINE).group(1))
        d1_mse = float(re.search(r'mseF      \(p2point\): (.+)', s, re.MULTILINE).group(1))
        d1_psnr = float(re.search(r'mseF,PSNR \(p2point\): (.+)', s, re.MULTILINE).group(1))
        d2_mse = float(re.search(r'mseF      \(p2plane\): (.+)', s, re.MULTILINE).group(1))
        d2_psnr = float(re.search(r'mseF,PSNR \(p2plane\): (.+)', s, re.MULTILINE).group(1))
    except AttributeError as e:
        print(s)
        raise e
    try:
        y_mse = float(re.search(r'c\[0\],    F         : (.+)', s, re.MULTILINE).group(1))
        u_mse = float(re.search(r'c\[1\],    F         : (.+)', s, re.MULTILINE).group(1))
        v_mse = float(re.search(r'c\[2\],    F         : (.+)', s, re.MULTILINE).group(1))
        y_mae = float(re.search(r'c\[0\],    maeF         : (.+)', s, re.MULTILINE).group(1))
        u_mae = float(re.search(r'c\[1\],    maeF         : (.+)', s, re.MULTILINE).group(1))
        v_mae = float(re.search(r'c\[2\],    maeF         : (.+)', s, re.MULTILINE).group(1))
        y_psnr = float(re.search(r'c\[0\],PSNRF         : (.+)', s, re.MULTILINE).group(1))
        u_psnr = float(re.search(r'c\[1\],PSNRF         : (.+)', s, re.MULTILINE).group(1))
        v_psnr = float(re.search(r'c\[2\],PSNRF         : (.+)', s, re.MULTILINE).group(1))
    except AttributeError:
        return {
            'd1_mae': d1_mae,
            'd1_mse': d1_mse,
            'd1_psnr': d1_psnr,
            'd2_mse': d2_mse,
            'd2_psnr': d2_psnr,
        }
    return {
        'd1_mae': d1_mae,
        'd1_mse': d1_mse,
        'd1_psnr': d1_psnr,
        'd2_mse': d2_mse,
        'd2_psnr': d2_psnr,
        'y_mse': y_mse,
        'u_mse': u_mse,
        'v_mse': v_mse,
        'y_mae': y_mae,
        'u_mae': u_mae,
        'v_mae': v_mae,
        'y_psnr': y_psnr,
        'u_psnr': u_psnr,
        'v_psnr': v_psnr
    }
