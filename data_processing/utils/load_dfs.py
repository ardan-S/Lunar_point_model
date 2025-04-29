import pandas as pd # type: ignore
import os
from pathlib import Path
import numpy as np
import requests
import matplotlib.pyplot as plt # type: ignore
import sys

from download_data import clear_dir
from utils.utils import (
    get_metadata_value, 
    clean_metadata_value, 
    parse_metadata_content, 
    plot_polar, 
    decode_image_file, 
    get_closest_channels, 
    save_by_lon_range, 
    create_hist, 
    from_csv_and_desc
    )


def process_LRO_image(image_file, address, metadata, data_type, max_val=1.0, min_val=0.0):
    # Extract image data
    file_path = Path(image_file)
    assert file_path.is_file(), f"File not found: {file_path}"

    file_extension = file_path.suffix[1:].lower()
    image_data = np.asarray(decode_image_file(file_path, file_extension, metadata, address))

    # Convert to scientific values
    scaling_factor = float(get_metadata_value(metadata, address, 'SCALING_FACTOR'))
    offset = float(get_metadata_value(metadata, address, 'OFFSET'))
    assert scaling_factor is not None or offset is not None, "Scaling factor and offset not found in metadata"

    missing_constants = {
        'LOLA': clean_metadata_value(metadata.get('MISSING_CONSTANT', -32768)),
        'MiniRF': clean_metadata_value(metadata.get('MISSING_CONSTANT', -1.7976931E+308)),
        'Diviner': clean_metadata_value(metadata.get('MISSING_CONSTANT', -32768))
    }

    missing_constant = missing_constants.get(data_type)
    assert missing_constant is not None, f"Missing constant not found for data type '{data_type}'"

    mask = (image_data != missing_constant)
    output_vals = np.where(mask, (image_data * scaling_factor) + offset, np.nan)    # Remove missing values BEFORE applying transform
    output_vals = np.where(output_vals > max_val, np.nan, output_vals)  # Remove extreme values
    output_vals = np.where(output_vals < min_val, np.nan, output_vals)

    return image_data, output_vals


def generate_LRO_coords(image_shape, metadata):
    lines, samples = image_shape

    projection_keys = ['LINE_PROJECTION_OFFSET', 'SAMPLE_PROJECTION_OFFSET', 'CENTER_LATITUDE',
                       'CENTER_LONGITUDE', 'MAP_RESOLUTION', 'MINIMUM_LATITUDE', 'MAXIMUM_LATITUDE', 'MAP_SCALE']
    line_proj_offset, sample_proj_offset, center_lat, center_lon, map_res, min_lat, max_lat, map_scale = \
        (float(get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', key)) for key in projection_keys)

    proj_type = get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'MAP_PROJECTION_TYPE', string=True)

    a_km = float(get_metadata_value(metadata, 'IMAGE_MAP_PROJECTION', 'A_AXIS_RADIUS'))  # Moon's radius in km
    a = 1737.4 * 1e3 if a_km is None else a_km * 1e3  # Default to Moon's radius if not found in metadata and convert to meters as map_scale is m/pixel

    if center_lat == 0.0:  # Equatorial (Mini-RF) - Simple Cylindrical
        assert proj_type == 'SIMPLE CYLINDRICAL', f"Unsupported projection type: {proj_type}"
        assert center_lat == 0.0 and center_lon == 0.0, "Center latitude and longitude must be 0.0 for Simple Cylindrical"

        line_idxs = np.arange(lines)
        sample_idxs = np.arange(samples)

        sample_grid, line_grid = np.meshgrid(sample_idxs, line_idxs)

        lats = (line_grid - line_proj_offset) / map_res
        if lats.max() > 91 or lats.min() < -91:
            raise ValueError(f"WARNING: Latitude values exceed bounds for Simple Cylindrical projection: {lats.min()} - {lats.max()}")

        lats = np.clip(lats, -90, 90)   # due to line projection offset, lats can reach +- 90.31 so clip is applied
        lons = (sample_grid - sample_proj_offset) / map_res

    else:   # Polar Stereographic (LOLA, Diviner)
        assert proj_type == 'POLAR STEREOGRAPHIC', f"Unsupported projection type: {proj_type}"

        # Generate pixel coordinates
        x = (np.arange(samples) - sample_proj_offset) * map_scale   
        y = (np.arange(lines) - line_proj_offset) * map_scale
        x, y = np.meshgrid(x, y)

        t = np.sqrt(x**2 + y**2)    # noqa for some operator warning
        c = 2 * np.arctan(t / (2 * a))

        lons = center_lon + np.degrees(np.arctan2(x, -y))

        if center_lat == 90.0:  # North Pole
            lats = center_lat - np.degrees(c)
        elif center_lat == -90.0:  # South Pole
            lats = center_lat + np.degrees(c)
        else:
            raise ValueError(f"Center latitude is not supported: {center_lat}")

    lons = lons % 360   # Convert to 0-360 range

    return lons, lats


def load_lro_df(data_dict, data_type, plot_frac=0.25, hist=False):

    if data_dict['save_path'] is not None:
        os.mkdir(data_dict['save_path']) if not os.path.exists(data_dict['save_path']) else None
    if data_dict['file_path'] is not None:
        os.mkdir(data_dict['file_path']) if not os.path.exists(data_dict['file_path']) else None
    if  data_dict['plot_path'] is not None:
        os.mkdir(data_dict['plot_path']) if not os.path.exists(data_dict['plot_path']) else None

    clear_dir(data_dict['save_path'])

    plot_save_path = data_dict['plot_path'] if 'plot_path' in data_dict else None

    if len([f for f in os.listdir(data_dict['save_path']) if f.endswith('.csv') and 'lon' in f]) == 12:
        print(f"Raw CSVs for {data_type} found at: {data_dict['save_path']}. Skipping load df...")

    else:
        print(f"Processing {data_type} data..."); sys.stdout.flush()

        file_path = data_dict['file_path']
        address = data_dict['address']
        lbl_ext = data_dict['lbl_ext']
        csv_save_path = data_dict['save_path'] if 'save_path' in data_dict else None
        max_val = data_dict['max']
        min_val = data_dict['min']

        assert csv_save_path or plot_save_path, "At least one of 'save_path' or 'plot_path' must be provided."
        assert isinstance(data_type, str), "data_type must be a string."

        lbl_files = [f for f in os.listdir(file_path) if f.endswith(lbl_ext)]

        if lbl_files == []:
            raise ValueError(f"No files found with extension '{lbl_ext}' in directory: {file_path}\nHave you downloaded the data?")

        for lbl_file in lbl_files:
            lbl_path = f"{file_path}/{lbl_file}"
            metadata = parse_metadata_content(lbl_path)

            img_file = lbl_path.replace(data_dict['lbl_ext'], data_dict['img_ext'])

            image_data, output_vals = process_LRO_image(img_file, address, metadata, data_type, max_val, min_val)
            output_vals = output_vals.flatten()

            lons, lats = generate_LRO_coords(image_data.shape, metadata)
            lons = lons.flatten()
            lats = lats.flatten()

            output_vals[(output_vals < min_val) | (output_vals > max_val)] = np.nan

            valid_mask = ((lats <= -75) & (lats >= -90)) | ((lats <= 90) & (lats >= 75))
            valid_mask &= np.isfinite(output_vals)  # Remove non-finite vals from output_vals and clip coords to poles

            assert np.all((lons >= 0) & (lons <= 360)), f"Some longitude values are out of bounds for {data_type} - Min: {lons.min()}, max: {lons.max()}\n{sum(lons<0)} \n{sum(lons>360)}"
            assert np.all((lats >= -90) & (lats <= 90)), f"Some latitude values are out of bounds for {data_type} - Min: {lats.min()}, max: {lats.max()}\n{sum(lats<-90)} \n{sum(lats>90)}"

            df = pd.DataFrame({
                'Longitude': lons[valid_mask],
                'Latitude': lats[valid_mask],
                data_type: output_vals[valid_mask]
            })

            assert data_type in df.columns, f"Data type '{data_type}' not found in dataframe columns."

            if csv_save_path:
                save_by_lon_range(df, csv_save_path)

            del df, image_data, output_vals, lons, lats

    if plot_save_path or hist:
        df = from_csv_and_desc(data_dict, data_type)
        print(f"Loaded {len(df):,} points for {data_type} data.")
    
        if plot_save_path:
            plot_polar(df, data_type, frac=plot_frac, save_path=plot_save_path, name_add='raw')
        if hist:
            create_hist(df, data_type)


def load_lola_df(data_dict, data_type, hist=False, plot_frac=0.25):

    if data_dict['save_path'] is not None:
        os.mkdir(data_dict['save_path']) if not os.path.exists(data_dict['save_path']) else None
    if data_dict['file_path'] is not None:
        os.mkdir(data_dict['file_path']) if not os.path.exists(data_dict['file_path']) else None
    if  data_dict['plot_path'] is not None:
        os.mkdir(data_dict['plot_path']) if not os.path.exists(data_dict['plot_path']) else None

    clear_dir(data_dict['save_path'])

    csv_save_path = data_dict['save_path'] if 'save_path' in data_dict else None
    plot_save_path = data_dict['plot_path'] if 'plot_path' in data_dict else None

    if len([f for f in os.listdir(data_dict['save_path']) if f.endswith('.csv') and 'lon' in f]) == 12:
        print(f"Raw CSVs for {data_type} found at: {data_dict['save_path']}. Skipping load df...")
    
    else:
        print(f"Processing {data_type} data..."); sys.stdout.flush()

        files = os.listdir(data_dict['file_path'])
        if len(files) == 0:
            raise ValueError(f"No files found in directory: {data_dict['file_path']}\nHave you downloaded the data?")
                
        filecount = 0
        skipped = 0
        for file in files:
            filecount += 1
            df_temp = pd.read_csv(os.path.join(data_dict['file_path'], file), sep=r'\s+', header=None)

            if df_temp.shape[1] < 3:
                raise ValueError(f"File {file} has only {df_temp.shape[1]} columns, expected at least 3. Processed {filecount} files.")

            df_temp = df_temp.iloc[:, :3]  # Select only the first three columns
            df_temp.columns = ['Longitude', 'Latitude', data_type]

            df_temp['Longitude'] = df_temp['Longitude'].astype(np.float32)
            df_temp['Latitude'] = df_temp['Latitude'].astype(np.float32)
            df_temp[data_type] = df_temp[data_type].astype(np.float32)

            # Check for values within latitude bounds
            points_in_bounds = ((df_temp['Latitude'] <= -75) & (df_temp['Latitude'] >= -90)) | ((df_temp['Latitude'] <= 90) & (df_temp['Latitude'] >= 75))
            if not points_in_bounds.any():
                skipped += 1
                continue

            max_val = data_dict['max']
            min_val = data_dict['min']

            # df_temp[data_type][(df_temp[data_type] < min_val) | (df_temp[data_type] > max_val)] = np.nan
            df_temp.loc[(df_temp[data_type] < min_val) | (df_temp[data_type] > max_val), data_type] = np.nan

            valid_mask = ((df_temp['Latitude'] <= -75) & (df_temp['Latitude'] >= -90)) | ((df_temp['Latitude'] <= 90) & (df_temp['Latitude'] >= 75))
            valid_mask &= np.isfinite(df_temp[data_type])  # Remove non-finite vals from output_vals and clip coords to poles

            assert np.all((df_temp['Longitude'] >= 0) & (df_temp['Longitude'] <= 360)), f"Some longitude values are out of bounds for {data_type}: \n{df_temp['Longitude']<0} \n{df_temp['Longitude']>360}"
            assert np.all((df_temp['Latitude'] >= -90) & (df_temp['Latitude'] <= 90)), f"Some latitude values are out of bounds for {data_type}: \n{df_temp['Latitude']<-90} \n{df_temp['Latitude']>90}"

            # Remove invalid values
            df_temp.drop_duplicates(subset=['Longitude', 'Latitude'], inplace=True)
            df_temp.reset_index(drop=True, inplace=True)

            if csv_save_path:
                save_by_lon_range(df_temp, csv_save_path)

        print(f"\nLOLA: {filecount} files processed out of {len(files)}, {skipped} files skipped due to no valid points.")

    if plot_save_path or hist:
        df = from_csv_and_desc(data_dict, data_type)

        if plot_save_path:
            plot_polar(df, data_type, frac=plot_frac, save_path=plot_save_path, name_add='raw')

        if hist:
            create_hist(df, data_type)


def process_M3_image(image_file, address, metadata):
    lines = int(get_metadata_value(metadata, address, 'LINES'))
    line_samples = int(get_metadata_value(metadata, address, 'LINE_SAMPLES'))
    bands = int(get_metadata_value(metadata, address, 'BANDS'))
    sample_bits = int(get_metadata_value(metadata, address, 'SAMPLE_BITS'))
    sample_type = str(get_metadata_value(metadata, address, 'SAMPLE_TYPE'))
    invalid_constant = clean_metadata_value(metadata.get('INVALID_CONSTANT', -999.0))

    target_wavelengths = [1300, 1500, 2000]     # Target wavelengths taken from Brown et al. (2022)
    test_channels = get_closest_channels(metadata, address, target_wavelengths)

    assert len(set(test_channels)) == len(test_channels), "Adjacent channels found in the closest channels list. Not supported."
    assert np.all((test_channels >= 1) & (test_channels <= bands)), "Channel index out of bounds"
    assert image_file.split('.')[-1].lower() == 'img', f"Unsupported file extension: {image_file.split('.')[-1].lower()}"

    if sample_type == 'PC_REAL' and sample_bits == 32:
        dtype = '<f4'   # Little-endian 32-bit float (as in M3 documentation)
    else:
        raise ValueError(f"Unsupported combination of SAMPLE_TYPE: {sample_type} and SAMPLE_BITS: {sample_bits}")

    with open(image_file, 'rb') as f:
        image_data = np.fromfile(f, dtype=dtype)

    if lines * line_samples * bands != image_data.size: # Handle cases where the image data is not the expected size
        extracted_bands = np.empty((lines, line_samples, len(test_channels)))
        output_vals = np.full((lines, line_samples), np.nan)
        return extracted_bands, output_vals


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

    extracted_bands = np.where((extracted_bands == invalid_constant) | (extracted_bands == 2 * invalid_constant), np.nan, extracted_bands)
    reference_bands = np.where((reference_bands == invalid_constant) | (reference_bands == 2 * invalid_constant), np.nan, reference_bands)
    troughs = np.where((extracted_bands < 1e-6) | (extracted_bands > 1.5), np.nan, extracted_bands)
    shoulders = np.where((reference_bands < 1e-6) | (reference_bands > 1.5), np.nan, reference_bands)

    BDRs = shoulders / (2 * troughs)    # Band depth ratio
    output_vals = np.min(BDRs, axis=2)    # Take the band with the minimum BDR for each point
    max = 1.75
    output_vals = np.clip(output_vals, None, max)  # Clip values to [0, max]

    return extracted_bands, output_vals


def generate_M3_coords(image_shape, metadata, data_dict):
    text_file = 'https://planetarydata.jpl.nasa.gov/img/data/m3/CH1M3_0003_md5.txt'
    response = requests.get(text_file)
    lines = response.text.splitlines()

    loc_file_dir = data_dict['file_path']
    loc_lbl_ext = data_dict['loc_lbl_ext']
    loc_img_ext = data_dict['loc_img_ext']

    loc_img_name = str(get_metadata_value(metadata, '', 'CH1:PIXEL_LOCATION_FILE_NAME', string=True))
    loc_lbl_name = loc_img_name.replace(loc_img_ext, loc_lbl_ext)

    loc_img_path = os.path.join(loc_file_dir, loc_img_name)
    loc_lbl_path = os.path.join(loc_file_dir, loc_lbl_name)

    assert os.path.isfile(loc_img_path), f"Location image file not found: {loc_img_path}"
    assert os.path.isfile(loc_lbl_path), f"Location label file not found: {loc_lbl_path}"

    loc_metadata = parse_metadata_content(loc_lbl_path)
    
    loc_address = data_dict['loc_address']

    lines = int(get_metadata_value(loc_metadata, loc_address, 'LINES'))
    line_samples = int(get_metadata_value(loc_metadata, loc_address, 'LINE_SAMPLES'))
    bands = int(get_metadata_value(loc_metadata, loc_address, 'BANDS'))
    sample_bits = int(get_metadata_value(loc_metadata, loc_address, 'SAMPLE_BITS'))
    sample_type = str(get_metadata_value(loc_metadata, loc_address, 'SAMPLE_TYPE'))

    if loc_lbl_path.split('.')[-1].lower() != 'lbl':
        raise ValueError("Unsupported file extension: {}".format(loc_lbl_path.split('.')[-1].lower()))

    dtype = {
        (32, 'PC_REAL'): np.float32,
        (64, 'PC_REAL'): np.float64
    }.get((sample_bits, sample_type))

    if dtype is None:
        raise ValueError(f"Unsupported combination of SAMPLE_TYPE: {sample_type} and SAMPLE_BITS: {sample_bits}")

    if lines is None or line_samples is None or bands is None:
        raise ValueError("Missing metadata values for lines, line_samples or bands")

    with open(loc_img_path, 'rb') as f:
        loc_data = np.fromfile(f, dtype='<f8')

    if loc_data.size != lines * line_samples * bands:
        raise ValueError(f"Mismatch in data size: expected {lines * line_samples * bands}, got {loc_data.size}")
    
    lons = np.empty((lines, line_samples))
    lats = np.empty((lines, line_samples))
    radii = np.empty((lines, line_samples))

    index = 0

    for i in range(lines):
        for arr in (lons, lats, radii):
            arr[i, :] = loc_data[index:index + line_samples]
            index += line_samples

    # Raise if any lon or lat values are out of bounds
    if not (np.all((0 <= lons) & (lons <= 360)) and np.all((-90 <= lats) & (lats <= 90))):
        raise ValueError("Some coordinate values are out of bounds.")

    reference_elevation = 1737400   # https://pds-imaging.jpl.nasa.gov/documentation/Isaacson_M3_Workshop_Final.pdf (pg.26, accessed 30/07/2024)
    elev = radii - reference_elevation

    return lons, lats, elev


def load_m3_df(data_dict, plot_frac=0.25, hist=False):

    if data_dict['save_path'] is not None:
        os.mkdir(data_dict['save_path']) if not os.path.exists(data_dict['save_path']) else None
    if data_dict['file_path'] is not None:
        os.mkdir(data_dict['file_path']) if not os.path.exists(data_dict['file_path']) else None
    if  data_dict['plot_path'] is not None:
        os.mkdir(data_dict['plot_path']) if not os.path.exists(data_dict['plot_path']) else None

    clear_dir(data_dict['save_path'])
    
    plot_save_path = data_dict['plot_path'] if 'plot_path' in data_dict else None

    if len([f for f in os.listdir(data_dict['save_path']) if f.endswith('.csv') and 'lon' in f]) == 12:
        print(f"Raw CSVs for M3 found at: {data_dict['save_path']}. Skipping load df...")
    
    else:
        print(f"Processing M3 data..."); sys.stdout.flush()

        file_path = data_dict['file_path']
        address = data_dict['address']
        lbl_ext = data_dict['lbl_ext']
        csv_save_path = data_dict['save_path'] if 'save_path' in data_dict else None

        assert plot_save_path or csv_save_path, "At least one of 'plot_path' or 'save_path' must be provided."

        lbl_files = [f for f in os.listdir(file_path) if f.endswith(lbl_ext)]
        if lbl_files == []:
            raise ValueError(f"No files found with extension '{lbl_ext}' in directory: {file_path}\nHave you downloaded the data?")

        for lbl_file in lbl_files:
            lbl_path = f"{file_path}/{lbl_file}"
            metadata = parse_metadata_content(lbl_path)
            img_file = lbl_path.replace(data_dict['lbl_ext'], data_dict['img_ext'])

            image_data, output_vals = process_M3_image(img_file, address, metadata)
            lons, lats, elev = generate_M3_coords(image_data.shape, metadata, data_dict)

            lons = lons.flatten()
            lats = lats.flatten()
            elev = elev.flatten()
            output_vals = output_vals.flatten()

            assert np.all((lons >= 0) & (lons <= 360)), "Some longitude values are out of bounds."
            assert np.all((lats >= -90) & (lats <= 90)), "Some latitude values are out of bounds."

            valid_mask = ((lats <= -75) & (lats >= -90)) | ((lats <= 90) & (lats >= 75))
            valid_mask &= np.isfinite(output_vals) & np.isfinite(elev)

            df = pd.DataFrame({
                'Longitude': lons[valid_mask],
                'Latitude': lats[valid_mask],
                'Elevation': elev[valid_mask],
                'M3': output_vals[valid_mask]
            })

            assert 'M3' in df.columns, f"Data type 'M3' not found in dataframe columns."

            if csv_save_path:
                save_by_lon_range(df, csv_save_path)

            del df, image_data, output_vals, lons, lats, elev

    if plot_save_path or hist:
        df = from_csv_and_desc(data_dict, 'M3')

        if plot_save_path:
            plot_polar(df, 'M3', frac=plot_frac, save_path=plot_save_path, name_add='raw')

        if hist: 
            create_hist(df, 'M3')

