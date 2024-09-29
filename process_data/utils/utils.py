import re
import requests
from collections import defaultdict
import glymur
import numpy as np
from osgeo import gdal
import pandas as pd
import os
from urllib.parse import urljoin
import io
from requests.exceptions import ChunkedEncodingError, ConnectionError
from http.client import IncompleteRead


def parse_metadata_content(file_content):
    if isinstance(file_content, bytes):
        try:
            content_str = file_content.decode('utf-8')
        except UnicodeDecodeError:
            print("Error decoding the file content with utf-8 encoding")
            return {}
    elif isinstance(file_content, str):
        content_str = file_content
    else:
        raise ValueError("Invalid file content type. Expected bytes or str")

    metadata = defaultdict(dict)
    object_stack = []

    try:
        current_object_path = ""
        for line in content_str.splitlines():
            line = line.strip()
            if line.startswith("OBJECT"):
                object_name = re.findall(r'OBJECT\s*=\s*(\w+)', line)[0]
                object_stack.append(object_name)
                current_object_path = '.'.join(object_stack)
                metadata[current_object_path] = {}
            elif line.startswith("END_OBJECT"):
                object_stack.pop()
                current_object_path = '.'.join(object_stack)
            elif "=" in line:
                key, value = map(str.strip, line.split('=', 1))
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.isdigit():
                    value = int(value)
                elif re.match(r'^\d+\.\d+$', value):
                    value = float(value)
                metadata[current_object_path][key] = value
    except UnicodeDecodeError:
        print("Error decoding the file content with utf-8 encoding")

    # Convert defaultdict to regular dict for compatibility
    return {k: dict(v) for k, v in metadata.items()}


def download_parse_metadata(url):
    response = requests.get(url)
    response.raise_for_status()
    file_content = response.content
    return parse_metadata_content(file_content)


def clean_metadata_value(value, string=False):
    if string:
        return str(value)
    try:
        cleaned_value = ''.join(filter(lambda x: x.isdigit() or x in ['.', '-'], str(value)))
        return float(cleaned_value)
    except ValueError:
        if value != 'PC_REAL':
            print(f"Error converting value to float: {value}")
        return str(value)


def get_metadata_value(metadata, object_path, key, string=False):
    return clean_metadata_value(metadata.get(object_path, {}).get(key), string=string)


def decode_image_file(file_path, file_extension, lines, line_samples, metadata, address):
    if file_extension == 'jp2':
        image_data = glymur.Jp2k(file_path)[:]

    # elif file_extension == 'img':
    #     bands = int(get_metadata_value(metadata, address, 'BANDS'))
    #     sample_bits = int(get_metadata_value(metadata, address, 'SAMPLE_BITS'))
    #     sample_type = str(get_metadata_value(metadata, address, 'SAMPLE_TYPE'))

    #     if sample_type == 'PC_REAL' and sample_bits == 32:
    #         dtype = np.float32
    #     else:
    #         raise ValueError(f"Unsupported combination of SAMPLE_TYPE: {sample_type} and SAMPLE_BITS: {sample_bits}")

    #     with open(file_path, 'rb') as f:
    #         image_data = np.fromfile(f, dtype=dtype)

    #         new_lines = int(lines)
    #         new_size = new_lines * line_samples * bands

    #         if new_size != image_data.size:
    #             raise ValueError(f"Mismatch in data size: expected {new_size}, got {image_data.size}")

    #         image_data = image_data.reshape((new_lines, line_samples, bands))

    #         if image_data.shape[-1] == 1:
    #             image_data = np.squeeze(image_data, axis=-1)
    #         else:
    #             raise ValueError(f"Unsupported number of bands: {bands}")
    # else:
    #     raise ValueError("Unsupported file extension: {}".format(file_extension))

    # return image_data

    elif file_extension == 'img':
        dataset = gdal.Open(file_path)
        if dataset is None:
            raise ValueError(f"Unable to open {file_path} with GDAL.")
        image_data = dataset.ReadAsArray()
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    return image_data


def get_closest_channels(metadata, address, target_wavelengths):
    def fetch_url(url, retries=3):
        if os.path.isfile(url):  # Check if the source is a local file
            with open(url, 'r') as file:
                return file.read()
        else:  # Assume the source is a URL
            for attempt in range(retries):
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    return response.text
                except (ChunkedEncodingError, ConnectionError, IncompleteRead) as e:
                    if attempt < retries - 1:
                        print(f'Attempt {attempt + 1} failed for url: {url}\n Error: {e}.\nRetrying...')
                        continue
                    else:
                        raise e

    base_calib_file = 'https://planetarydata.jpl.nasa.gov/img/data/m3/CH1M3_0003/CALIB/'
    calib_file = str(get_metadata_value(metadata, '', 'CH1:SPECTRAL_CALIBRATION_FILE_NAME', string=True))
    calib_file_url = urljoin(base_calib_file, calib_file)

    response = fetch_url(calib_file_url)

    calib_data = pd.read_csv(io.StringIO(response), sep=r'\s+', header=None, names=["Channel", "Wavelength"])
    calib_data["Channel"] = calib_data["Channel"].astype(int)
    calib_data["Wavelength"] = calib_data["Wavelength"].astype(float)

    # Find the closest channel for each target wavelength
    closest_channels = []
    for wavelength in target_wavelengths:
        idx = (calib_data['Wavelength'] - wavelength).abs().argmin()
        channel = calib_data.iloc[idx]['Channel']
        closest_channels.append(channel)

    test_channels = np.array(closest_channels).astype(int)

    return test_channels


def plot_df(df, plot_save_path):
    raise NotImplementedError("plot_df function is not implemented yet")
