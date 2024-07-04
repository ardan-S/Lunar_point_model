import numpy as np
import pandas as pd
import glymur
import tempfile
import requests
from collections import defaultdict
import re
from urllib.parse import urljoin
import io


# done
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


# done
def download_parse_metadata(url):
    response = requests.get(url)
    response.raise_for_status()
    file_content = response.content
    return parse_metadata_content(file_content)


# done
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


# done
def get_metadata_value(metadata, object_path, key, string=False):
    return clean_metadata_value(metadata.get(object_path, {}).get(key), string=string)


# done
def extract_LRO_image(image_url, address, metadata=None, fraction_read=1.0):
    lines = get_metadata_value(metadata, address, 'LINES')
    line_samples = int(get_metadata_value(metadata, address, 'LINE_SAMPLES'))

    if fraction_read != 1.0:
        print(f"Downloading only {fraction_read * 100}% of the image\n")
        response = requests.head(image_url)
        file_size = int(response.headers['content-length'])
        line_size = file_size / lines
        num_lines_downloaded = int(lines * fraction_read)
        bytes_to_download = int(line_size * num_lines_downloaded)
        headers = {'Range': f'bytes=0-{bytes_to_download-1}'}
        response = requests.get(image_url, headers=headers)

    else:
        response = requests.get(image_url)

    response.raise_for_status()

    file_extension = image_url.split('.')[-1].lower()

    with tempfile.NamedTemporaryFile(suffix=f".{file_extension}") as temp_file:
        temp_file.write(response.content)
        temp_file.flush()

        if file_extension == 'jp2':
            image_data = glymur.Jp2k(temp_file.name)[:]

        elif file_extension == 'img':
            bands = int(get_metadata_value(metadata, address, 'BANDS'))
            sample_bits = int(get_metadata_value(metadata, address, 'SAMPLE_BITS'))
            sample_type = str(get_metadata_value(metadata, address, 'SAMPLE_TYPE'))

            if sample_type == 'PC_REAL' and sample_bits == 32:
                dtype = np.float32
            else:
                raise ValueError(f"Unsupported combination of SAMPLE_TYPE: {sample_type} and SAMPLE_BITS: {sample_bits}")

            with open(temp_file.name, 'rb') as f:
                image_data = np.fromfile(f, dtype=dtype)

                new_lines = int(lines * fraction_read)
                new_size = new_lines * line_samples * bands

                if new_size != image_data.size:
                    raise ValueError(f"Mismatch in data size: expected {new_size}, got {image_data.size}")

                image_data = image_data.reshape((new_lines, line_samples, bands))

                if image_data.shape[-1] == 1:
                    image_data = np.squeeze(image_data, axis=-1)
                else:
                    raise ValueError(f"Unsupported number of bands: {bands}")
        else:
            raise ValueError("Unsupported file extension: {}".format(file_extension))
    return image_data


# done
def extract_M3_image(image_url, metadata):
    address = 'RFL_FILE.RFL_IMAGE'
    lines = int(get_metadata_value(metadata, address, 'LINES'))
    line_samples = int(get_metadata_value(metadata, address, 'LINE_SAMPLES'))
    bands = int(get_metadata_value(metadata, address, 'BANDS'))
    sample_bits = int(get_metadata_value(metadata, address, 'SAMPLE_BITS'))
    sample_type = str(get_metadata_value(metadata, address, 'SAMPLE_TYPE'))

    base_calib_file = 'https://planetarydata.jpl.nasa.gov/img/data/m3/CH1M3_0003/CALIB/'
    calib_file = str(get_metadata_value(metadata, '', 'CH1:SPECTRAL_CALIBRATION_FILE_NAME', string=True))
    calib_file_url = urljoin(base_calib_file, calib_file)

    response = requests.get(calib_file_url)
    response.raise_for_status()

    # Read the .TAB file into a pandas DataFrame
    calib_data = pd.read_csv(io.StringIO(response.text), sep=r'\s+', header=None, names=["Channel", "Wavelength"])
    calib_data["Channel"] = calib_data["Channel"].astype(int)
    calib_data["Wavelength"] = calib_data["Wavelength"].astype(float)
    target_wavelengths = [1300, 1500, 2000]
    closest_channels = [calib_data.iloc[(calib_data['Wavelength'] - wavelength).abs().argmin()]['Channel'] for wavelength in target_wavelengths]
    test_channels = np.array(closest_channels).astype(int)
    ref_channels = np.where(np.isin(test_channels + 1, test_channels), test_channels - 1, test_channels + 1)

    if np.any(np.isin(ref_channels, test_channels)):
        raise ValueError("Could not find a suitable ref_channels value")

    # Read the image file into a numpy array
    response = requests.get(image_url)
    response.raise_for_status()

    if image_url.split('.')[-1].lower() != 'img':
        raise ValueError("Unsupported file extension: {}".format(image_url.split('.')[-1].lower()))

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
            if (lines * line_samples * bands) != image_data.size:
                raise ValueError(f"Mismatch in data size: expected {lines * line_samples * bands}, got {image_data.size}")

            image_data = image_data.reshape((lines, line_samples, bands))
            extracted_bands = image_data[:, :, test_channels - 1]  # -1 to convert to 0-based index
            reference_bands = image_data[:, :, ref_channels - 1]  # -1 to convert to 0-based index

    return extracted_bands, reference_bands


# done
def process_LRO_image(image_data, metadata, address, data_type):
    scaling_factor = get_metadata_value(metadata, address, 'SCALING_FACTOR')
    offset = get_metadata_value(metadata, address, 'OFFSET')
    if scaling_factor is None or offset is None:
        raise ValueError(f"Scaling factor and/or offset not found in metadata for data type '{data_type}'")

    # Define missing constants
    missing_constants = {
        'LOLA': clean_metadata_value(metadata.get('IMAGE_MISSING_CONSTANT', -32768)),
        'MiniRF': clean_metadata_value(metadata.get('MISSING_CONSTANT', -1.7976931E+308)),
        'Diviner': clean_metadata_value(metadata.get('MISSING_CONSTANT', -32768))
    }

    missing_constant = missing_constants.get(data_type, missing_constants['Diviner'])   # Default to Diviner

    mask = (image_data != missing_constant)
    output_vals = np.where(mask, (image_data * scaling_factor) + offset, np.nan)

    return output_vals


# !!!!!!!!!!!!!!!!!!!!
# CONFIRM THIS
# !!!!!!!!!!!!!!!!!!
def process_M3_image(image_data, ref_data):
    ratios = ref_data / image_data
    min_vals = np.min(ratios, axis=2)    # Take the min val as if this is above threshold, they all are.
    print(f'Shape of min_vals: {min_vals.shape}')
    # Number of values outside of 0 and 1.2 in each of the test channels
    nums_outside = np.sum((min_vals < 0) | (min_vals > 1.2))
    print(f'Number of values outside of 0 and 1.2 in each of the test channels: {nums_outside} out of {min_vals.size} ({(nums_outside/min_vals.size)*100:.2f}%)')
    output_vals = np.where((min_vals < 0) | (min_vals > 1.2), np.nan, min_vals)    # Check if any of the ratios are outside the range
    return output_vals


# done
def generate_LRO_coords(image_shape, metadata):
    lines, samples = image_shape
    projection_keys = ['LINE_PROJECTION_OFFSET', 'SAMPLE_PROJECTION_OFFSET', 'CENTER_LATITUDE',
                       'CENTER_LONGITUDE', 'MAP_RESOLUTION', 'MINIMUM_LATITUDE', 'MAXIMUM_LATITUDE']
    line_proj_offset, sample_proj_offset, center_lat, center_lon, map_res, min_lat, max_lat = \
        (get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', key) for key in projection_keys)
    a = 1737.4  # Moon's radius in km

    # Generate pixel coordinates and do the Polar Stereographic projection
    x = (np.arange(samples) - sample_proj_offset) / map_res
    y = (np.arange(lines) - line_proj_offset) / map_res
    x, y = np.meshgrid(x, y)
    t = np.sqrt(x**2 + y**2)
    c = 2 * np.arctan(t / (2 * a))

    if center_lat == 90.0:  # North Pole
        lats = center_lat - np.degrees(c)
    elif center_lat == -90.0:  # South Pole
        lats = center_lat + np.degrees(c)
    elif center_lat == 0.0:  # Equatorial (Mini-RF)
        lats = np.degrees(np.arcsin(y / a))
    else:
        raise ValueError(f"Center latitude is not supported: {center_lat}")

    # Adjust latitude range to min_lat to max_lat
    lat_scale = (max_lat - min_lat) / (np.max(lats) - np.min(lats))
    lats = min_lat + (lats - np.min(lats)) * lat_scale
    lons = center_lon + np.degrees(np.arctan2(y, x))

    lats = np.clip(lats, min_lat, max_lat)
    lons = (center_lon + lons) % 360

    return lons, lats


# done
def generate_M3_coords(image_shape, metadata):
    base_loc_file = 'https://planetarydata.jpl.nasa.gov/img/data/m3/CH1M3_0003/DATA/20081118_20090214/200811/L1B/'
    loc_file = str(get_metadata_value(metadata, '', 'CH1:PIXEL_LOCATION_FILE_NAME', string=True))
    loc_file_url = urljoin(base_loc_file, loc_file)
    loc_lbl_file = loc_file_url.replace('LOC.IMG', 'L1B.LBL')
    loc_metadata = download_parse_metadata(loc_lbl_file)
    loc_address = 'LOC_FILE.LOC_IMAGE'

    lines = int(get_metadata_value(loc_metadata, loc_address, 'LINES'))
    line_samples = int(get_metadata_value(loc_metadata, loc_address, 'LINE_SAMPLES'))
    bands = int(get_metadata_value(loc_metadata, loc_address, 'BANDS'))
    sample_bits = int(get_metadata_value(loc_metadata, loc_address, 'SAMPLE_BITS'))
    sample_type = str(get_metadata_value(loc_metadata, loc_address, 'SAMPLE_TYPE'))

    response = requests.get(loc_file_url)
    response.raise_for_status()

    if loc_file_url.split('.')[-1].lower() != 'img':
        raise ValueError("Unsupported file extension: {}".format(loc_file_url.split('.')[-1].lower()))

    dtype = {
        (32, 'PC_REAL'): np.float32,
        (64, 'PC_REAL'): np.float64
    }.get((sample_bits, sample_type))

    if dtype is None:
        raise ValueError(f"Unsupported combination of SAMPLE_TYPE: {sample_type} and SAMPLE_BITS: {sample_bits}")

    with tempfile.NamedTemporaryFile(suffix=".img") as temp_file:
        temp_file.write(response.content)
        temp_file.flush()
        loc_data = np.fromfile(temp_file.name, dtype=dtype).reshape((lines, line_samples, bands))

    if loc_data.size != lines * line_samples * bands:
        raise ValueError(f"Mismatch in data size: expected {lines * line_samples * bands}, got {loc_data.size}")

    lons, lats, radii = loc_data[:, :, 0], loc_data[:, :, 1], loc_data[:, :, 2]

    return lons, lats, radii


# done
def optimize_df(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df


# done
def process_image(metadata, image_path, data_type, output_csv_path=None):
    accepted_data_types = ['Diviner', 'LOLA', 'M3', 'MiniRF']
    if data_type not in accepted_data_types:
        raise ValueError(f"Invalid data type '{data_type}'. Accepted values are: {accepted_data_types}")

    if data_type == 'M3':
        image_data, ref_data = extract_M3_image(image_path, metadata)
        output_vals = process_M3_image(image_data, ref_data)
        lons, lats, _ = generate_M3_coords(image_data.shape, metadata)
    else:
        address = 'IMAGE' if data_type == 'MiniRF' else 'UNCOMPRESSED_FILE.IMAGE'
        image_data = extract_LRO_image(image_path, address, metadata, 1) if data_type == 'MiniRF' else extract_LRO_image(image_path, address, metadata)
        output_vals = process_LRO_image(image_data, metadata, address, data_type)
        lons, lats = generate_LRO_coords(image_data.shape, metadata)

    df = pd.DataFrame({
        'Longitude': lons.flatten(),
        'Latitude': lats.flatten(),
        data_type: output_vals.flatten()
    })

    df = filter_and_optimize_df(df, data_type, metadata)

    return df


# done
def filter_and_optimize_df(df, data_type, metadata):
    df = df.loc[((df['Latitude'] >= -90) & (df['Latitude'] <= -75)) | ((df['Latitude'] >= 75) & (df['Latitude'] <= 90))].copy()
    if data_type == 'MiniRF':
        df = MiniRF_sense_check(df)
    elif data_type == 'LOLA':
        df = LOLA_sense_check(df)
    elif data_type == 'Diviner':
        df = Diviner_sense_check(df, metadata)
    elif data_type == 'M3':
        df = M3_sense_check(df)
    return optimize_df(df)


# done
def Diviner_sense_check(df, metadata):
    max_temp = clean_metadata_value(metadata.get('DERIVED_MAXIMUM', 400))
    min_temp = clean_metadata_value(metadata.get('DERIVED_MINIMUM', 0))
    df['Diviner'] = np.clip(df['Diviner'], min_temp, max_temp)
    return df


# done
def LOLA_sense_check(df):
    df.loc[(df['LOLA'] < 0) | (df['LOLA'] > 1), 'LOLA'] = np.nan
    return df


# done
def MiniRF_sense_check(df):
    df.loc[(df['MiniRF'] < 0) | (df['MiniRF'] > 2), 'MiniRF'] = np.nan
    return df


# done
def M3_sense_check(df):
    df.loc[df['M3'] < 0, 'M3'] = np.nan
    return df
