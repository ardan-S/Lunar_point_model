import numpy as np
import pandas as pd
import glymur
import tempfile
import requests
from collections import defaultdict
import re


def download_parse_metadata(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    file_content = response.content
    metadata = defaultdict(dict)
    object_stack = []

    enc = 'utf-8'
    try:
        content_str = file_content.decode(enc)
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
        print(f"Error decoding the file content with encoding {enc}")

    # Convert defaultdict to regular dict for compatibility
    return {k: dict(v) for k, v in metadata.items()}


def clean_metadata_value(value):
    try:
        cleaned_value = ''.join(filter(lambda x: x.isdigit() or x in ['.', '-'], str(value)))
        return float(cleaned_value)
    except ValueError:
        print(f"Error converting value to float: {value}")
        return str(value)


def get_metadata_value(metadata, object_path, key):
    return clean_metadata_value(metadata.get(object_path, {}).get(key))


def parse_envi_header(header_file_content):
    header_info = {}
    lines = header_file_content.split('\n')
    for line in lines:
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('{}').strip()
            if ',' in value:
                value = [v.strip() for v in value.split(',')]
            header_info[key] = value
    return header_info


def extract_image_data(image_url, data_type, address, metadata=None, fraction_read=1.0):
    if fraction_read != 1.0:
        print(f"Downloading only {fraction_read * 100}% of the image\n")
        response = requests.head(image_url)
        file_size = int(response.headers['content-length'])
        lines = int(get_metadata_value(metadata, address, 'LINES'))
        line_samples = int(get_metadata_value(metadata, address, 'LINE_SAMPLES'))
        print(f'lines: {lines}, line_samples: {line_samples}')
        line_size = file_size / lines
        num_lines_downloaded = int(lines * fraction_read)
        bytes_to_download = int(line_size * num_lines_downloaded)
        headers = {'Range': f'bytes=0-{bytes_to_download-1}'}
        response = requests.get(image_url, headers=headers)

    else:
        lines = get_metadata_value(metadata, address, 'LINES')
        line_samples = get_metadata_value(metadata, address, 'LINE_SAMPLES')
        response = requests.get(image_url)

    response.raise_for_status()

    file_extension = image_url.split('.')[-1].lower()

    if file_extension == 'jp2':
        with tempfile.NamedTemporaryFile(suffix=".jp2") as temp_file:
            temp_file.write(response.content)
            temp_file.flush()  # Ensure all data is written to the file
            jp2 = glymur.Jp2k(temp_file.name)
            image_data = jp2[:]

    elif file_extension == 'img':
        bands = int(get_metadata_value(metadata, address, 'BANDS'))
        sample_bits = int(get_metadata_value(metadata, address, 'SAMPLE_BITS'))
        sample_type = str(get_metadata_value(metadata, address, 'SAMPLE_TYPE'))
        print(f"address: {address}, bands: {bands}, sample_bits: {sample_bits}, sample_type: {sample_type}")

        # Determine the numpy dtype based on SAMPLE_TYPE and SAMPLE_BITS
        if sample_type == 'PC_REAL' and sample_bits == 32:
            dtype = np.float32
        else:
            raise ValueError(f"Unsupported combination of SAMPLE_TYPE: {sample_type} and SAMPLE_BITS: {sample_bits}")

        with tempfile.NamedTemporaryFile(suffix=".img") as temp_file:
            temp_file.write(response.content)
            temp_file.flush()  # Ensure all data is written to the file
            with open(temp_file.name, 'rb') as f:
                image_data = np.fromfile(f, dtype=dtype)

                new_lines = int(lines * fraction_read)
                new_size = new_lines * line_samples * bands

                if new_size != image_data.size:
                    raise ValueError(f"Mismatch in data size: expected {new_size}, got {image_data.size}")

                image_data = image_data.reshape((new_lines, line_samples, bands))

                if image_data.shape[-1] == 1:
                    image_data = np.squeeze(image_data, axis=-1)
                    print(f"Condensed image shape: {image_data.shape}")
                else:
                    raise ValueError(f"Unsupported number of bands: {bands}")
    else:
        raise ValueError("Unsupported file extension: {}".format(file_extension))
    return image_data


def convert_dn_to_val(dn, scaling_factor, offset, missing_constant):
    val = np.full(dn.shape, np.nan)
    mask = (dn != missing_constant)
    val[mask] = (dn[mask] * scaling_factor) + offset
    return val


# def generate_coordinates(image_shape, metadata):
#     lines, samples = image_shape
#     line_projection_offset = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'LINE_PROJECTION_OFFSET')
#     sample_projection_offset = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'SAMPLE_PROJECTION_OFFSET')
#     center_lat = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'CENTER_LATITUDE')
#     center_lon = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'CENTER_LONGITUDE')
#     map_resolution = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'MAP_RESOLUTION')
#     min_lat = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'MINIMUM_LATITUDE')
#     max_lat = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'MAXIMUM_LATITUDE')

#     # Assuming MAP_RESOLUTION is in degrees per pixel
#     deg_per_pixel = 1.0 / map_resolution

#     lats = center_lat - ((np.arange(lines) - line_projection_offset) * deg_per_pixel)
#     lons = center_lon + ((np.arange(samples) - sample_projection_offset) * deg_per_pixel)

#     # Ensure lons and lats fall within specified min/max values
#     lats = np.clip(lats, min_lat, max_lat)
#     lons = np.mod(lons, 360.0)

#     return np.meshgrid(lons, lats)


def generate_lat_lon_arrays(image_shape, metadata):
    lines, samples = image_shape
    line_projection_offset = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'LINE_PROJECTION_OFFSET')
    sample_projection_offset = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'SAMPLE_PROJECTION_OFFSET')
    center_lat = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'CENTER_LATITUDE')
    center_lon = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'CENTER_LONGITUDE')
    map_resolution = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'MAP_RESOLUTION')
    min_lat = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'MINIMUM_LATITUDE')
    max_lat = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'MAXIMUM_LATITUDE')
    a = 1737.4  # Moon's radius in km

    print(f'sample projection offset: {sample_projection_offset}, line projection offset: {line_projection_offset}, samples: {samples}, lines: {lines}, map resolution: {map_resolution}')
    print(f"center_lat: {center_lat}, center_lon: {center_lon}, map_resolution: {map_resolution}, min_lat: {min_lat}, max_lat: {max_lat}")

    # Generate pixel coordinate arrays
    x = (np.arange(samples) - sample_projection_offset) / map_resolution
    y = (np.arange(lines) - line_projection_offset) / map_resolution
    x, y = np.meshgrid(x, y)

    # Polar Stereographic projection formula
    t = np.sqrt(x**2 + y**2)
    c = 2 * np.arctan(t / (2 * a))

    if center_lat == 90.0:  # North Pole
        lats = center_lat - np.degrees(c)
    elif center_lat == -90.0:  # South Pole
        lats = center_lat + np.degrees(np.pi/2 - c)
    elif center_lat == 0.0:  # Equatorial (Mini-RF)
        lats = np.degrees(np.arcsin(y / a))
    else:
        raise ValueError(f"Center latitude is not supported: {center_lat}")

    print(f'Lats min: {lats.min()}, max: {lats.max()}')

    # Adjust latitude range to min_lat to max_lat
    lat_scale = (max_lat - min_lat) / (np.max(lats) - np.min(lats))
    lats = min_lat + (lats - np.min(lats)) * lat_scale

    lons = center_lon + np.degrees(np.arctan2(y, x))

    print(f'Lats min: {lats.min()}, max: {lats.max()}, Lons min: {lons.min()}, max: {lons.max()}')

    # Adjust ranges
    lats = np.clip(lats, min_lat, max_lat)
    lons = (center_lon + lons) % 360

    return lons, lats


def save_to_csv(lons, lats, julian_dates, output_csv_path):
    data = {
        'Longitude': lons.flatten(),
        'Latitude': lats.flatten(),
        'Julian Date': julian_dates.flatten(),
    }

    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"Data saved to {output_csv_path}")


def optimize_df(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df


def process_image(metadata, image_path, data_type, output_csv_path=None):

    accepted_data_types = ['date', 'Diviner', 'LOLA', 'M3-data', 'M3-loc', 'MiniRF']
    if data_type not in accepted_data_types:
        raise ValueError(f"Invalid data type '{data_type}'. Accepted values are: {accepted_data_types}")

    if data_type == 'M3-data':
        address = 'RFL_FILE.RFL_IMAGE'
    elif data_type == 'MiniRF':
        address = 'IMAGE'
    else:
        address = 'UNCOMPRESSED_FILE.IMAGE'

    print('Extracting image data...')
    image_data = (extract_image_data(image_path, data_type, address, metadata, 0.03) if data_type == 'MiniRF' else extract_image_data(image_path, data_type, address, metadata))
    print('Image data extracted')

    scaling_factor = get_metadata_value(metadata, address, 'SCALING_FACTOR')
    offset = get_metadata_value(metadata, address, 'OFFSET')
    if scaling_factor is None or offset is None:
        raise ValueError(f"Scaling factor and/or offset not found in metadata for data type '{data_type}'")

    if data_type == 'LOLA':
        missing_constant = (metadata.get('IMAGE_MISSING_CONSTANT', -32768))
    elif data_type == 'MiniRF':
        missing_constant = clean_metadata_value(metadata.get('MISSING_CONSTANT', -np.inf))
    else:
        missing_constant = clean_metadata_value(metadata.get('MISSING_CONSTANT', -32768))   # Default missing constant for Diviner

    lons, lats = generate_lat_lon_arrays(image_data.shape, metadata)
    print('Coordinates generated')
    output_vals = convert_dn_to_val(image_data, scaling_factor, offset, missing_constant)

    if output_csv_path:
        save_to_csv(lons, lats, output_vals, output_csv_path)

    df = pd.DataFrame({
        'Longitude': lons.flatten(),
        'Latitude': lats.flatten(),
        data_type: output_vals.flatten(),
    })

    # Remove lats outside the ranges of [75, 90] and [-90, -75]
    # df = df[(df['Latitude'] >= -90) & (df['Latitude'] <= -75) | (df['Latitude'] >= 75) & (df['Latitude'] <= 90)]

    df = MiniRF_sense_check(df) if data_type == 'MiniRF' else df
    df = LOLA_sense_check(df) if data_type == 'LOLA' else df
    df = Diviner_sense_check(df, metadata) if data_type == 'Diviner' else df
    print(f'Number of Lats between -85 and -90 (1): {len(df[(df["Latitude"] >= -90) & (df["Latitude"] <= -85)])}')
    return optimize_df(df)


def Diviner_sense_check(df, metadata):
    max_temp = clean_metadata_value(metadata.get('DERIVED_MAXIMUM', 400))
    min_temp = clean_metadata_value(metadata.get('DERIVED_MINIMUM', 0))
    df['Diviner'] = np.clip(df['Diviner'], min_temp, max_temp)
    return df


def LOLA_sense_check(df):
    """
    CHOSEN TO REMOVE VALUES OUT OF BANDS RATHER THAN CLIPPING THEM
    """
    df['LOLA'] = np.where((df['LOLA'] < 0) | (df['LOLA'] > 1), np.nan, df['LOLA'])

    return df


def MiniRF_sense_check(df):
    """
    CHOSEN TO REMOVE VALUES OUT OF BANDS RATHER THAN CLIPPING THEM
    """
    df['MiniRF'] = np.where((df['MiniRF'] < 0) | (df['MiniRF'] > 2), np.nan, df['MiniRF'])
    return df
