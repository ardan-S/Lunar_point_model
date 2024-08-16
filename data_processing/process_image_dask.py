import numpy as np
import pandas as pd
import glymur
import tempfile
import requests
from collections import defaultdict
import re
from urllib.parse import urljoin
import io
from requests.exceptions import ChunkedEncodingError, ConnectionError
from urllib3.exceptions import IncompleteRead
import sys
import os
from pathlib import Path
from collections import Counter

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
def is_url(path):
    return path.startswith('http://') or path.startswith('https://')


# done
def extract_LRO_image(image_url, address, metadata=None, fraction_read=1.0):

    lines = get_metadata_value(metadata, address, 'LINES')
    line_samples = int(get_metadata_value(metadata, address, 'LINE_SAMPLES'))

    if is_url(image_url):
        if fraction_read != 1.0:
            assert 0 < fraction_read <= 1, "Fraction read must be between 0 and 1"
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
            file_path = temp_file.name
            image_data = process_image_file(file_path, file_extension, lines, line_samples, metadata, address, fraction_read)

    else:
        file_path = Path(image_url)
        if not file_path.is_file():
            raise ValueError(f"File not found: {file_path}")
        file_extension = file_path.suffix[1:].lower()
        image_data = process_image_file(file_path, file_extension, lines, line_samples, metadata, address, fraction_read)

    return image_data


# Done
def process_image_file(file_path, file_extension, lines, line_samples, metadata, address, fraction_read):
    if file_extension == 'jp2':
        image_data = glymur.Jp2k(file_path)[:]

    elif file_extension == 'img':
        bands = int(get_metadata_value(metadata, address, 'BANDS'))
        sample_bits = int(get_metadata_value(metadata, address, 'SAMPLE_BITS'))
        sample_type = str(get_metadata_value(metadata, address, 'SAMPLE_TYPE'))

        if sample_type == 'PC_REAL' and sample_bits == 32:
            dtype = np.float32
        else:
            raise ValueError(f"Unsupported combination of SAMPLE_TYPE: {sample_type} and SAMPLE_BITS: {sample_bits}")

        with open(file_path, 'rb') as f:
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

    def fetch_url(url, retries=3):
        if os.path.isfile(url):  # Check if the source is a local file
            with open(url, 'r') as file:
                return file.read()
        else:  # Assume the source is a URL
            for attempt in range (retries):
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    return response
                except (ChunkedEncodingError, ConnectionError, IncompleteRead) as e:
                    if attempt < retries - 1:
                        print(f'Attempt {attempt + 1} failed for url: {url}\n Error: {e}.\nRetrying...')
                        continue
                    else:
                        raise e

    address = 'RFL_FILE.RFL_IMAGE'
    lines = int(get_metadata_value(metadata, address, 'LINES'))
    line_samples = int(get_metadata_value(metadata, address, 'LINE_SAMPLES'))
    bands = int(get_metadata_value(metadata, address, 'BANDS'))
    sample_bits = int(get_metadata_value(metadata, address, 'SAMPLE_BITS'))
    sample_type = str(get_metadata_value(metadata, address, 'SAMPLE_TYPE'))
    invalid_constant = clean_metadata_value(metadata.get('INVALID_CONSTANT', -999.0))

    base_calib_file = 'https://planetarydata.jpl.nasa.gov/img/data/m3/CH1M3_0003/CALIB/'
    calib_file = str(get_metadata_value(metadata, '', 'CH1:SPECTRAL_CALIBRATION_FILE_NAME', string=True))
    calib_file_url = urljoin(base_calib_file, calib_file)

    response = fetch_url(calib_file_url)

    calib_data = pd.read_csv(io.StringIO(response.text), sep=r'\s+', header=None, names=["Channel", "Wavelength"])
    calib_data["Channel"] = calib_data["Channel"].astype(int)
    calib_data["Wavelength"] = calib_data["Wavelength"].astype(float)

    target_wavelengths = [1300, 1500, 2000]     # Target wavelengths taken from Brown et al. (2022)
    closest_channels = [calib_data.iloc[(calib_data['Wavelength'] - wavelength).abs().argmin()]['Channel'] for wavelength in target_wavelengths]
    test_channels = np.array(closest_channels).astype(int)

    del calib_data, response

    if len(set(closest_channels)) < len(closest_channels):  # Check for adjacent channels
        raise ValueError("Adjacent channels found in the closest channels list. Not supported.")

    if np.any(test_channels < 1) or np.any(test_channels > bands):  # Check for out of bounds channels
        raise ValueError("Channel index out of bounds")

    if image_url.split('.')[-1].lower() != 'img':   # Check for valid file extension
        raise ValueError("Unsupported file extension: {}".format(image_url.split('.')[-1].lower()))

    if sample_type == 'PC_REAL' and sample_bits == 32:
        dtype = '<f4'   # Little-endian 32-bit float (as in M3 documentation)
    else:
        raise ValueError(f"Unsupported combination of SAMPLE_TYPE: {sample_type} and SAMPLE_BITS: {sample_bits}")

    with open(image_url, 'rb') as f:
        image_data = np.fromfile(f, dtype=dtype)

    if (lines * line_samples * bands) != image_data.size:   # Check for mismatch in data size
        raise ValueError(f"Mismatch in data size: expected {lines * line_samples * bands}, got {image_data.size}")

    # Define bands
    extracted_bands = np.empty((lines, line_samples, len(test_channels)))
    reference_band_up = np.full((lines, line_samples, len(test_channels)), np.nan)
    reference_band_down = np.full((lines, line_samples, len(test_channels)), np.nan)

    for idx, channel in enumerate(test_channels):
        channel_start_idx = channel - 1    # Convert to 0-based index
        
        for i in range(lines):
            start_idx = (i * bands + channel_start_idx) * line_samples
            end_idx = start_idx + line_samples
            extracted_bands[i, :, idx] = image_data[start_idx:end_idx]

            up_start_idx = (i * bands + channel_start_idx + 1) * line_samples
            up_end_idx = up_start_idx + line_samples
            reference_band_up[i, :, idx] = image_data[up_start_idx:up_end_idx]

            down_start_idx = (i * bands + channel_start_idx - 1) * line_samples
            down_end_idx = down_start_idx + line_samples
            reference_band_down[i, :, idx] = image_data[down_start_idx:down_end_idx]

    del image_data

    # Element-wise operation to handle NaNs and sum the bands
    reference_bands = np.where(np.isnan(reference_band_up), 2 * reference_band_down, 
                                np.where(np.isnan(reference_band_down), 2 * reference_band_up, 
                                        reference_band_up + reference_band_down))
    
    # Remove invalid and out of range values
    extracted_bands = np.where((extracted_bands == invalid_constant) | (extracted_bands == 2 * invalid_constant), np.nan, extracted_bands)
    reference_bands = np.where((reference_bands == invalid_constant) | (reference_bands == 2 * invalid_constant), np.nan, reference_bands)
    extracted_bands = np.where((extracted_bands < 1e-6) | (extracted_bands > 1.5), np.nan, extracted_bands)
    reference_bands = np.where((reference_bands < 1e-6) | (reference_bands > 1.5), np.nan, reference_bands)

    return extracted_bands, reference_bands


# done
def process_LRO_image(image_data, metadata, address, data_type):
    address = 'COMPRESSED_FILE' if data_type == 'LOLA' else address
    scaling_factor = get_metadata_value(metadata, address, 'SCALING_FACTOR')
    offset = get_metadata_value(metadata, address, 'OFFSET')
    if scaling_factor is None or offset is None:
        raise ValueError(f"Scaling factor and/or offset not found in metadata for data type '{data_type}'")

    # Define missing constants - named slightly differently in metadata for each data type
    missing_constants = {
        'LOLA': clean_metadata_value(metadata.get('IMAGE_MISSING_CONSTANT', -32768)),
        'MiniRF': clean_metadata_value(metadata.get('MISSING_CONSTANT', -1.7976931E+308)),
        'Diviner': clean_metadata_value(metadata.get('MISSING_CONSTANT', -32768))
    }

    missing_constant = missing_constants.get(data_type, missing_constants['Diviner'])   # Default to Diviner

    mask = (image_data != missing_constant)
    output_vals = np.where(mask, (image_data * scaling_factor) + offset, np.nan)    # Remove missing values BEFORE applying transform
    output_vals = np.where(output_vals > 1e10, np.nan, output_vals)
    return output_vals


def process_M3_image(trough, shoulder):
    BDRs = shoulder / (2*trough)    # Band depth ratio
    output_vals = np.min(BDRs, axis=2)    # Take the band with the minimum BDR for each point
    # output_vals = np.max(BDRs, axis=2)    # Take the band with the maximum BDR for each point
    max = 1.75
    output_vals = np.clip(output_vals, None, max)  # Clip values to [0, max]

    print(f"\nNumber of calc vals above {max} before clip: {np.sum(output_vals > max)} out of {output_vals.size} ({((np.sum(output_vals > max)) / output_vals.size) *100:.2f}%)")
    print(f"Range of values: {np.nanmin(output_vals):.3f} to {np.nanmax(output_vals):.3f}")
    print(f"1st percentile: {np.nanpercentile(output_vals, 1):.3f}, 99th percentile: {np.nanpercentile(output_vals, 99):.3f}")
    print(f"25th percentile: {np.nanpercentile(output_vals, 25):.3f}, 75th percentile: {np.nanpercentile(output_vals, 75):.3f}")
    return output_vals


# done
def generate_LRO_coords(image_shape, metadata, data_type):
    lines, samples = image_shape
    projection_keys = ['LINE_PROJECTION_OFFSET', 'SAMPLE_PROJECTION_OFFSET', 'CENTER_LATITUDE',
                       'CENTER_LONGITUDE', 'MAP_RESOLUTION', 'MINIMUM_LATITUDE', 'MAXIMUM_LATITUDE']
    line_proj_offset, sample_proj_offset, center_lat, center_lon, map_res, min_lat, max_lat = \
        (get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', key) for key in projection_keys)
    proj_type = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'MAP_PROJECTION_TYPE', string=True)
    a = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'A_AXIS_RADIUS') if center_lat == 0.0 else 1737.4   # Moon's radius in km

    # Generate pixel coordinates
    x = (np.arange(samples) - sample_proj_offset) / map_res
    y = (np.arange(lines) - line_proj_offset) / map_res 
    x, y = np.meshgrid(x, y)

    if center_lat == 0.0:  # Equatorial (Mini-RF) - Simple Cylindrical
        assert proj_type == 'SIMPLE CYLINDRICAL', f"Unsupported projection type: {proj_type}"
        assert center_lat == 0.0 and center_lon == 0.0, "Center latitude and longitude must be 0.0 for Simple Cylindrical"
        lons = np.degrees(x / a)
        print(f"Range of lons before adjustment: {np.min(lons)} to {np.max(lons)}")
        lon_scale = 360 / (2 * np.max(np.abs(lons)))  # Scale factor for longitudes
        lons = (lons * lon_scale + 360) % 360  # Apply scaling and wrap to [0, 360)
        print(f"Range of lons after adjustment: {np.min(lons)} to {np.max(lons)}")

        # Map y directly to latitude range [-90, 90]
        lats = y * (max_lat - min_lat) / np.max(np.abs(y))  # Scale y to latitude range

    else:   # Polar Stereographic (LOLA, Diviner)
        assert proj_type == 'POLAR STEREOGRAPHIC', f"Unsupported projection type: {proj_type}"
        t = np.sqrt(x**2 + y**2)
        c = 2 * np.arctan(t / (2 * a))

        lons = center_lon + np.degrees(np.arctan2(y, x))
        lons = (center_lon + lons) % 360

        if center_lat == 90.0:  # North Pole
            lats = center_lat - np.degrees(c)
        elif center_lat == -90.0:  # South Pole
            lats = center_lat + np.degrees(c)
        else:
            raise ValueError(f"Center latitude is not supported: {center_lat}")

    # Adjust latitude range to min_lat to max_lat
    lat_scale = (max_lat - min_lat) / (np.max(lats) - np.min(lats))
    lats = min_lat + (lats - np.min(lats)) * lat_scale
    lats = np.clip(lats, min_lat, max_lat)

    return lons, lats


# done
def generate_M3_coords(image_shape, metadata):
    text_file = 'https://planetarydata.jpl.nasa.gov/img/data/m3/CH1M3_0003_md5.txt'
    response = requests.get(text_file)
    lines = response.text.splitlines()

    loc_file_dir = Path(os.getenv('RDS')) / 'ephemeral' / 'as5023' / 'M3' / 'raw_files'
    loc_lbl_urls = [os.path.join(loc_file_dir, f) for f in os.listdir(loc_file_dir) if f.endswith('_L1B.LBL')]
    loc_img_urls = [os.path.join(loc_file_dir, f) for f in os.listdir(loc_file_dir) if f.endswith('LOC.IMG')]
    loc_img_name = str(get_metadata_value(metadata, '', 'CH1:PIXEL_LOCATION_FILE_NAME', string=True))
    loc_lbl_name = loc_img_name.replace('LOC.IMG', 'L1B.LBL')

    loc_lbl_file = None
    loc_img_file = None

    # Loop through the lines and find the one that contains the loc_file_name
    for loc_lbl_url in loc_lbl_urls:
        if loc_lbl_name in loc_lbl_url:
            loc_lbl_file = loc_lbl_url
            break

    # Loop through lines in the text file and find the one that contains the loc_img_name
    for loc_img_url in loc_img_urls:
        if loc_img_name in loc_img_url:
            loc_img_file = loc_img_url
            break

    if not loc_lbl_file or not loc_img_file:
        raise ValueError(f"Location file not found for {loc_lbl_name} or {loc_img_name}")

    with open(loc_lbl_url, 'rb') as f:
        loc_metadata = parse_metadata_content(f.read())
    loc_address = 'LOC_FILE.LOC_IMAGE'

    lines = int(get_metadata_value(loc_metadata, loc_address, 'LINES'))
    line_samples = int(get_metadata_value(loc_metadata, loc_address, 'LINE_SAMPLES'))
    bands = int(get_metadata_value(loc_metadata, loc_address, 'BANDS'))
    sample_bits = int(get_metadata_value(loc_metadata, loc_address, 'SAMPLE_BITS'))
    sample_type = str(get_metadata_value(loc_metadata, loc_address, 'SAMPLE_TYPE'))

    if loc_lbl_url.split('.')[-1].lower() != 'lbl':
        raise ValueError("Unsupported file extension: {}".format(loc_lbl_url.split('.')[-1].lower()))

    dtype = {
        (32, 'PC_REAL'): np.float32,
        (64, 'PC_REAL'): np.float64
    }.get((sample_bits, sample_type))

    if dtype != np.float64:
        raise ValueError(f"Unsupported combination of SAMPLE_TYPE: {sample_type} and SAMPLE_BITS: {sample_bits}")
    
    if lines is None or line_samples is None or bands is None:
        raise ValueError("Missing metadata values for lines, line_samples or bands")

    with open(loc_img_url, 'rb') as f:
        loc_data = np.fromfile(f, dtype='<f8')

    if loc_data.size != lines * line_samples * bands:
        raise ValueError(f"Mismatch in data size: expected {lines * line_samples * bands}, got {loc_data.size}")

    # Initialize empty arrays for longitude, latitude, and radii
    lons = np.empty((lines, line_samples))
    lats = np.empty((lines, line_samples))
    radii = np.empty((lines, line_samples))

    index = 0

    for i in range(lines):
        for arr in (lons, lats, radii):
            arr[i, :] = loc_data[index:index + line_samples]
            index += line_samples

    # Raise if any lon or lat values are out of bounds
    if np.any(lons < 0) or np.any(lons > 360) or np.any(lats < -90) or np.any(lats > 90):
        raise ValueError(f"Some coordinate values are out of bounds.\nMin/max for lon, lat: {np.min(lons)}, {np.max(lons)}, {np.min(lats)}, {np.max(lats)}")

    reference_elevation = 1737400   # https://pds-imaging.jpl.nasa.gov/documentation/Isaacson_M3_Workshop_Final.pdf (pg.26, accessed 30/07/2024)
    elev = radii - reference_elevation

    del loc_data, loc_metadata
    return lons, lats, elev


# done
def optimize_df(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df.loc[:, col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df.loc[:, col] = pd.to_numeric(df[col], downcast='integer')
    return df


# done
def process_image(metadata, image_path, data_type, output_csv_path=None):
    accepted_data_types = ['Diviner', 'LOLA', 'M3', 'MiniRF']
    if data_type not in accepted_data_types:
        raise ValueError(f"Invalid data type '{data_type}'. Accepted values are: {accepted_data_types}")

    if data_type == 'M3':
        image_data, ref_data = extract_M3_image(image_path, metadata)
        output_vals = process_M3_image(image_data, ref_data)
        lons, lats, elev = generate_M3_coords(image_data.shape, metadata)
    else:
        address = 'IMAGE' if data_type == 'MiniRF' else 'UNCOMPRESSED_FILE.IMAGE'
        image_data = extract_LRO_image(image_path, address, metadata)
        output_vals = process_LRO_image(image_data, metadata, address, data_type)
        lons, lats = generate_LRO_coords(image_data.shape, metadata, data_type)


    df = pd.DataFrame({
        'Longitude': lons.flatten(),
        'Latitude': lats.flatten(),
        data_type: output_vals.flatten()
    })

    if data_type == 'M3':
        df['Elevation'] = elev.flatten()

    # Raise if error with coordinate generation
    if np.any(df['Longitude'] < 0) or np.any(df['Longitude'] > 360):
        raise ValueError("Some longitude values are out of bounds")
    
    if np.any(df['Latitude'] < -90) or np.any(df['Latitude'] > 90):
        raise ValueError("Some latitude values are out of bounds")

    return filter_and_optimize_df(df, data_type, metadata)


# done
def filter_and_optimize_df(df, data_type, metadata):
    # Ensure all Lat values are in the ranges of either [-75, -90] or [75, 90]
    df = df[(df['Latitude'] <= -75) & (df['Latitude'] >= -90) | (df['Latitude'] <= 90) & (df['Latitude'] >= 75)]

    sense_check_functions = {   
    'MiniRF': MiniRF_sense_check,
    'LOLA': LOLA_sense_check,
    'Diviner': lambda df: Diviner_sense_check(df, metadata),
    'M3': M3_sense_check
    }
    
    if data_type in sense_check_functions:
        df = sense_check_functions[data_type](df)
    else:
        raise ValueError(f"Sense check function not found for data type '{data_type}'")
    
    return optimize_df(df)


# done
def Diviner_sense_check(df, metadata):
    max_temp = clean_metadata_value(metadata.get('DERIVED_MAXIMUM', 400))
    min_temp = clean_metadata_value(metadata.get('DERIVED_MINIMUM', 0))
    df['Diviner'] = np.clip(df['Diviner'], min_temp, max_temp)
    return df


# done
def LOLA_sense_check(df):
    df.loc[(df['LOLA'] < 0) | (df['LOLA'] > 2), 'LOLA'] = np.nan
    return df


# done
def MiniRF_sense_check(df):
    df.loc[(df['MiniRF'] < 0) | (df['MiniRF'] > 5), 'MiniRF'] = np.nan
    df.loc[df['MiniRF'] > 1.5, 'MiniRF'] = 1.5
    return df


# done
def M3_sense_check(df):
    df = df.copy()
    df.loc[df['M3'] < 0, 'M3'] = np.nan
    return df
