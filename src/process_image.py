import numpy as np
import pandas as pd

import glymur
import tempfile
import requests


def download_parse_metadata(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    file_content = response.content
    metadata = {}
    enc = 'utf-8'
    try:
        content_str = file_content.decode(enc)
        for line in content_str.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"')
                metadata[key] = value
    except UnicodeDecodeError:
        print(f"Error decoding the file content with encoding {enc}")
    return metadata


def extract_jp2_data(jp2_url):
    response = requests.get(jp2_url)
    response.raise_for_status()  # Ensure we notice bad responses

    with tempfile.NamedTemporaryFile(suffix=".jp2") as temp_file:
        temp_file.write(response.content)
        temp_file.flush()  # Ensure all data is written to the file
        jp2 = glymur.Jp2k(temp_file.name)
        image_data = jp2[:]

    return image_data


def convert_dn_to_val(dn, scaling_factor, offset, missing_constant):
    val = np.full(dn.shape, np.nan)
    mask = (dn != missing_constant)
    val[mask] = (dn[mask] * scaling_factor) + offset
    return val


def clean_metadata_value(value):
    try:
        cleaned_value = ''.join(filter(lambda x: x.isdigit() or x in ['.', '-'], value))
        return float(cleaned_value)
    except ValueError:
        print(f"Error converting metadata value to float: {value}")
        return None


def generate_coordinates(image_shape, metadata):
    lines, samples = image_shape
    line_projection_offset = clean_metadata_value(metadata['LINE_PROJECTION_OFFSET'])
    sample_projection_offset = clean_metadata_value(metadata['SAMPLE_PROJECTION_OFFSET'])
    center_lat = clean_metadata_value(metadata['CENTER_LATITUDE'])
    center_lon = clean_metadata_value(metadata['CENTER_LONGITUDE'])
    map_resolution = clean_metadata_value(metadata['MAP_RESOLUTION'])

    # Assuming MAP_RESOLUTION is in degrees per pixel
    deg_per_pixel = 1.0 / map_resolution

    lats = center_lat - ((np.arange(lines) - line_projection_offset) * deg_per_pixel)
    lons = center_lon + ((np.arange(samples) - sample_projection_offset) * deg_per_pixel)

    # Ensure lons and lats fall within specified min/max values
    # lats = np.clip(lats, clean_metadata_value(metadata['MINIMUM_LATITUDE']), clean_metadata_value(metadata['MAXIMUM_LATITUDE']))
    lons = np.mod(lons, 360.0)

    return np.meshgrid(lons, lats)


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


def process_image(metadata, jp2_file_path, output_csv_path, data_type):

    accepted_data_types = ['date', 'temp', 'LOLA']
    if data_type not in accepted_data_types:
        raise ValueError(f"Invalid data type '{data_type}'. Accepted values are: {accepted_data_types}")

    image_data = extract_jp2_data(jp2_file_path)

    scaling_factor = clean_metadata_value(metadata.get('SCALING_FACTOR', 1))  # Default to 1 if not found
    offset = clean_metadata_value(metadata.get('OFFSET', 0))  # Default to 0 if not found
    missing_constant = (metadata.get('IMAGE_MISSING_CONSTANT', -32768))

    lons, lats = generate_coordinates(image_data.shape, metadata)

    # if data_type == 'date':
    #     output_vals = convert_dn_to_val(image_data, scaling_factor, offset, missing_constant)

    # elif data_type == 'temp':
    #     output_vals = convert_dn_to_val(image_data, scaling_factor, offset, missing_constant)

    output_vals = convert_dn_to_val(image_data, scaling_factor, offset, missing_constant)

    df = pd.DataFrame({
        'Longitude': lons.flatten(),
        'Latitude': lats.flatten(),
        data_type: output_vals.flatten(),
    })

    return optimize_df(df)
