import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import dask.dataframe as dd
import os
from filelock import FileLock
import gc

import aiohttp
import asyncio

from process_image_dask import process_image, parse_metadata_content


async def fetch(session, url):
    """
    Asynchronous function to fetch the content of a URL using aiohttp.
    """
    async with session.get(url) as response:
        return await response.text()


async def get_file_urls_async(page_url, file_extension, keyword, limit=None):
    """
    Asynchronously retrieve file URLs from a webpage that match the specified extension and keyword.
    """
    async with aiohttp.ClientSession() as session:
        html = await fetch(session, page_url)
        soup = BeautifulSoup(html, 'html.parser')

        urls = [
            urljoin(page_url, anchor.get('href'))
            for anchor in soup.find_all('a')
            if anchor.get('href') and anchor.get('href').endswith(file_extension) and keyword in anchor.get('href')
        ]

        return urls[:limit] if limit else urls


async def get_M3_urls_async(page_url, file_extension, keyword):
    """
    Asynchronously retrieve M3 file URLs from a metadata page
    """
    async with aiohttp.ClientSession() as session:
        response = await fetch(session, page_url)
        base_url = page_url.rsplit('/', 1)[0] + '/'

        urls = [
            urljoin(base_url, parts[1])
            for line in response.splitlines()
            if (parts := line.split()) and len(parts) == 2 and parts[1].endswith(file_extension) and keyword in parts[1]
        ]

        return urls


async def fetch_metadata(source):
    """
    Asynchronously fetch metadata content from a local file or URL.
    """
    if os.path.isfile(source):  # Check if the source is a local file
        with open(source, 'r') as file:
            return file.read()
    else:  # Assume the source is a URL
        async with aiohttp.ClientSession() as session:
            async with session.get(source) as response:
                response.raise_for_status()
                return await response.text()


async def download_parse_metadata_async(url):
    """
    Asynchronously download and parse metadata from a URL.
    """
    content = await fetch_metadata(url)
    return parse_metadata_content(content)


async def process_image_async(metadata, image_url, label_type, output_csv_path):
    """
    Asynchronously process an image.
    """
    return process_image(metadata, image_url, label_type, output_csv_path)


async def process_single_url(url_info):
    """
    Asynchronously process a single URL by downloading the metadata, processing the image and returning the result.
    """
    label_url, image_url, output_csv_path, label_type = url_info
    metadata = await download_parse_metadata_async(label_url)
    df = await process_image_async(metadata, image_url, label_type, output_csv_path)
    return df


def get_file_urls(page_url, file_extension, keyword, limit=None):
    """
    Retrieve file URLs from a webpage that match the specified extension and keyword.
    """
    response = requests.get(page_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')

    urls = [
        urljoin(page_url, anchor.get('href'))
        for anchor in soup.find_all('a')
        if anchor.get('href') and anchor.get('href').endswith(file_extension) and keyword in anchor.get('href')
    ]

    return urls[:limit] if limit else urls


def get_M3_urls(page_url, file_extension, keyword):
    """
    Retrieve M3 file URLs from a metadata page
    """
    response = requests.get(page_url)
    response.raise_for_status()

    base_url = page_url.rsplit('/', 1)[0] + '/'

    urls = [
        urljoin(base_url, parts[1])
        for line in response.text.splitlines()
        if (parts := line.split()) and len(parts) == 2 and parts[1].endswith(file_extension) and keyword in parts[1]
    ]

    return urls


def construct_image_url(url, label_type):
    """
    Construct the corresponding image URL from a label URL based on the label type.
    """
    replacements = {
        'LOLA': ('_jp2.lbl', '.jp2'),
        'M3': ('_L2.LBL', '_RFL.IMG'),
        'MiniRF': ('.lbl', '.img'),
        'Diviner': ('.lbl', '.jp2')
    }
    return url.replace(*replacements.get(label_type, ('.lbl', '.jp2')))


def process_urls_in_parallel(client, lbl_urls, data_type, output_dir):
    """
    Process a list of URLs in parallel using Dask, saving the results to CSV files.
    """
    url_info_list = [(lbl_url, construct_image_url(lbl_url, data_type), output_dir, data_type) for lbl_url in lbl_urls]
    
    # Create directory for CSVs if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    lon_ranges = [(0, 30), (30, 60), (60, 90),
                  (90, 120), (120, 150), (150, 180),
                  (180, 210), (210, 240), (240, 270),
                  (270, 300), (300, 330), (330, 360)]
    file_names = [
        os.path.join(output_dir, 'lon_000_030.csv'),
        os.path.join(output_dir, 'lon_030_060.csv'),
        os.path.join(output_dir, 'lon_060_090.csv'),
        os.path.join(output_dir, 'lon_090_120.csv'),
        os.path.join(output_dir, 'lon_120_150.csv'),
        os.path.join(output_dir, 'lon_150_180.csv'),
        os.path.join(output_dir, 'lon_180_210.csv'),
        os.path.join(output_dir, 'lon_210_240.csv'),
        os.path.join(output_dir, 'lon_240_270.csv'),
        os.path.join(output_dir, 'lon_270_300.csv'),
        os.path.join(output_dir, 'lon_300_330.csv'),
        os.path.join(output_dir, 'lon_330_360.csv')
    ]

    # Initialise CSVs with headers if they don't exist
    for file_name in file_names:
        if not os.path.isfile(file_name):
            with open (file_name, 'w') as f:
                if data_type == 'M3':
                    f.write(f'Longitude,Latitude,{data_type},Elevation\n')
                else:
                    f.write(f'Longitude,Latitude,{data_type}\n')

    # Submit tasks to the Dask scheduler
    futures = [
        client.submit(process_single_url, url_info)
        for url_info in url_info_list
    ]

    for future in futures:
        result_df = future.result()
        result_df = result_df.dropna()
        if result_df.shape[0] > 0:
            for (start, end), file_name in zip(lon_ranges, file_names):
                filtered_df = result_df[(result_df['Longitude'] >= start) & (result_df['Longitude'] < end)]
                if not filtered_df.empty:
                    print(f"Details for longitude range {start} - {end}:")
                    print(filtered_df.describe())
                    filtered_df.to_csv(file_name, mode='a', header=False, index=False)

                else:
                    if data_type != 'M3':
                        print(f'No data for longitude range {start} - {end}')

            # After the entire df is saved to CSV, remove from memory
            del result_df
            gc.collect()
    
    print('Files saved into respective CSVs')
