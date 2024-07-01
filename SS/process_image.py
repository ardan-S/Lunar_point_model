import numpy as np
import pandas as pd
import glymur
import tempfile
import requests
from collections import defaultdict
import re
from urllib.parse import urljoin
import io


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


def clean_metadata_value(value, string=False):
    if string:
        return str(value)
    try:
        cleaned_value = ''.join(filter(lambda x: x.isdigit() or x in ['.', '-'], str(value)))
        return float(cleaned_value)
    except ValueError:
        if value != ('PC_REAL'):
            print(f"Error converting value to float: {value}")
        return str(value)


def get_metadata_value(metadata, object_path, key, string=False):
    return clean_metadata_value(metadata.get(object_path, {}).get(key), string=string)


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
    tab_file = io.StringIO(response.text)
    calib_data = pd.read_csv(tab_file, sep='\s+', header=None, names=["Channel", "Wavelength"])

    calib_data["Channel"] = calib_data["Channel"].astype(int)
    calib_data["Wavelength"] = calib_data["Wavelength"].astype(float)
    target_wavelengths = [1300, 1500, 2000]

    def find_closest_channel(df, target_wavelength):
        return df.iloc[(df['Wavelength'] - target_wavelength).abs().argmin()]

    closest_channels = [find_closest_channel(calib_data, wavelength)['Channel'] for wavelength in target_wavelengths]
    test_channels = np.array(closest_channels).astype(int)
    ref_channels = test_channels + 1
    # Check if a value from one_band_higher is present in test_channels
    if np.any(np.isin(ref_channels, test_channels)):
        ref_channels = test_channels - 1
    if np.any(np.isin(ref_channels, test_channels)):
        raise ValueError("Could not find a suitable ref_channels value")

    # Read the image file into a numpy array
    response = requests.get(image_url)
    response.raise_for_status()

    file_extension = image_url.split('.')[-1].lower()
    if file_extension != 'img':
        raise ValueError("Unsupported file extension: {}".format(file_extension))

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

            new_size = lines * line_samples * bands
            if new_size != image_data.size:
                raise ValueError(f"Mismatch in data size: expected {new_size}, got {image_data.size}")

            image_data = image_data.reshape((lines, line_samples, bands))
            extracted_bands = image_data[:, :, test_channels - 1]  # -1 to convert to 0-based index
            reference_bands = image_data[:, :, ref_channels - 1]  # -1 to convert to 0-based index

    return extracted_bands, reference_bands


# !! REWRITE !!
def process_LRO_image(image_data, metadata, address, data_type):
    scaling_factor = get_metadata_value(metadata, address, 'SCALING_FACTOR')
    offset = get_metadata_value(metadata, address, 'OFFSET')
    if scaling_factor is None or offset is None:
        raise ValueError(f"Scaling factor and/or offset not found in metadata for data type '{data_type}'")

    if data_type == 'LOLA':
        missing_constant = (metadata.get('IMAGE_MISSING_CONSTANT', -32768))
    elif data_type == 'MiniRF':
        missing_constant = clean_metadata_value(metadata.get('MISSING_CONSTANT', -np.inf))  # !!!!! CHANGE THIS !!!!!!!
    else:
        missing_constant = clean_metadata_value(metadata.get('MISSING_CONSTANT', -32768))   # Default missing constant for Diviner

    def convert_dn_to_val(dn, scaling_factor, offset, missing_constant):
        val = np.full(dn.shape, np.nan)
        mask = (dn != missing_constant)
        val[mask] = (dn[mask] * scaling_factor) + offset
        return val

    output_vals = convert_dn_to_val(image_data, scaling_factor, offset, missing_constant)
    return output_vals


# !!!!!!!!!!!!!!!!!!!!
# CONFIRM THIS
# !!!!!!!!!!!!!!!!!!
def process_M3_image(image_data, ref_data, metadata):
    ratios = ref_data/image_data
    output_vals = np.min(ratios, axis=2)    # Take the min val as if this is above threshold, they all are.
    return output_vals


def generate_lat_lon_arrays(image_shape, metadata, data_type):
    if data_type == 'M3-data':
        lines, samples, _ = image_shape
    else:
        lines, samples = image_shape
    line_projection_offset = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'LINE_PROJECTION_OFFSET')
    sample_projection_offset = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'SAMPLE_PROJECTION_OFFSET')
    center_lat = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'CENTER_LATITUDE')
    center_lon = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'CENTER_LONGITUDE')
    map_resolution = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'MAP_RESOLUTION')
    min_lat = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'MINIMUM_LATITUDE')
    max_lat = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'MAXIMUM_LATITUDE')
    a = 1737.4  # Moon's radius in km

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

    # Adjust latitude range to min_lat to max_lat
    lat_scale = (max_lat - min_lat) / (np.max(lats) - np.min(lats))
    lats = min_lat + (lats - np.min(lats)) * lat_scale

    lons = center_lon + np.degrees(np.arctan2(y, x))

    print(f'Lats min: {lats.min()}, max: {lats.max()}, Lons min: {lons.min()}, max: {lons.max()}')

    # Adjust ranges
    lats = np.clip(lats, min_lat, max_lat)
    lons = (center_lon + lons) % 360

    return lons, lats


def generate_M3_lat_lon_arrays(image_shape, metadata):
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

    file_extension = loc_file_url.split('.')[-1].lower()
    if file_extension != 'img':
        raise ValueError("Unsupported file extension: {}".format(file_extension))

    # Determine the numpy dtype based on SAMPLE_TYPE and SAMPLE_BITS
    if sample_type == 'PC_REAL' and sample_bits == 32:
        dtype = np.float32
    elif sample_type == 'PC_REAL' and sample_bits == 64:
        dtype = np.float64
    else:
        raise ValueError(f"Unsupported combination of SAMPLE_TYPE: {sample_type} and SAMPLE_BITS: {sample_bits}")

    with tempfile.NamedTemporaryFile(suffix=".img") as temp_file:
        temp_file.write(response.content)
        temp_file.flush()
        with open(temp_file.name, 'rb') as f:
            loc_data = np.fromfile(f, dtype=dtype)

            new_size = lines * line_samples * bands
            if new_size != loc_data.size:
                raise ValueError(f"Mismatch in data size: expected {new_size}, got {loc_data.size}")

            loc_data = loc_data.reshape((lines, line_samples, bands))
            lons = loc_data[:, :, 0]
            lats = loc_data[:, :, 1]
            radii = loc_data[:, :, 2]

    return lons, lats, radii


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

    if data_type == 'M3-data':
        print('Processing M3 image data...')
        image_data, ref_data = extract_M3_image(image_path, metadata)
        print('Image data extracted')
        output_vals = process_M3_image(image_data, ref_data, metadata)
        print('Values processed')
        lons, lats, radii = generate_M3_lat_lon_arrays(image_data.shape, metadata)
        print('Coordinates generated')
    else:
        print('Processing LRO image data...')
        image_data = (extract_LRO_image(image_path, address, metadata, 0.04) if data_type == 'MiniRF' else extract_LRO_image(image_path, address, metadata))
        print('Image data extracted')
        output_vals = process_LRO_image(image_data, metadata, address, data_type)
        print('Values processed')
        lons, lats = generate_lat_lon_arrays(image_data.shape, metadata, data_type)
        print('Coordinates generated')

    df = pd.DataFrame({
        'Longitude': lons.flatten(),
        'Latitude': lats.flatten(),
        data_type: output_vals.flatten(),
    })

    # Remove lats outside the ranges of [75, 90] and [-90, -75]
    df = df[(df['Latitude'] >= -90) & (df['Latitude'] <= -75) | (df['Latitude'] >= 75) & (df['Latitude'] <= 90)]

    df = MiniRF_sense_check(df) if data_type == 'MiniRF' else df
    df = LOLA_sense_check(df) if data_type == 'LOLA' else df
    df = Diviner_sense_check(df, metadata) if data_type == 'Diviner' else df
    df = M3_sense_check(df) if data_type == 'M3-data' else df

    if output_csv_path:
        df.to_csv(output_csv_path, index=False)
        print(f'CSV saved to {output_csv_path}')

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


def M3_sense_check(df):
    df['M3-data'] = np.where((df['M3-data'] < 0), np.nan, df['M3-data'])
    return df
