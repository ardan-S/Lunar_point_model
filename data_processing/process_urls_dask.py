import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import dask.dataframe as dd
import os

import aiohttp
import asyncio

from process_image_dask import process_image, parse_metadata_content


async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()


async def get_file_urls_async(page_url, file_extension, keyword, limit=None):
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
    async with aiohttp.ClientSession() as session:
        response = await fetch(session, page_url)
        base_url = page_url.rsplit('/', 1)[0] + '/'

        urls = [
            urljoin(base_url, parts[1])
            for line in response.splitlines()
            if (parts := line.split()) and len(parts) == 2 and parts[1].endswith(file_extension) and keyword in parts[1]
        ]

        return urls


async def fetch_metadata(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.text()


async def download_parse_metadata_async(url):
    content = await fetch_metadata(url)
    return parse_metadata_content(content)


async def process_image_async(metadata, image_url, label_type, output_csv_path):
    return process_image(metadata, image_url, label_type, output_csv_path)


async def process_single_url(url_info):
    label_url, image_url, output_csv_path, label_type = url_info
    metadata = await download_parse_metadata_async(label_url)
    df = await process_image_async(metadata, image_url, label_type, output_csv_path)
    return df


def get_file_urls(page_url, file_extension, keyword, limit=None):
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
    replacements = {
        'LOLA': ('_JP2.LBL', '.JP2'),
        'M3': ('_L2.LBL', '_RFL.IMG'),
        'MiniRF': ('.lbl', '.img'),
        'Diviner': ('.lbl', '.jp2')
    }
    return url.replace(*replacements.get(label_type, ('.lbl', '.jp2')))


def process_urls_in_parallel(client, urls, label_type, output_dir):
    url_info_list = [(url, construct_image_url(url, label_type), output_dir, label_type) for url in urls]
    
    # Create directory for CSVs if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    lon_ranges = [(0, 60), (60, 120), (120, 180), (180, 240), (240, 300), (300, 360)]
    file_names = [
        os.path.join(output_dir, 'lon_0_60.csv'),
        os.path.join(output_dir, 'lon_60_120.csv'),
        os.path.join(output_dir, 'lon_120_180.csv'),
        os.path.join(output_dir, 'lon_180_240.csv'),
        os.path.join(output_dir, 'lon_240_300.csv'),
        os.path.join(output_dir, 'lon_300_360.csv')
    ]

    # Initialise CSVs with headers if they don't exist
    for file_name in file_names:
        if not os.path.isfile(file_name):
            with open (file_name, 'w') as f:
                f.write(f'Longitude,Latitude,{label_type}\n')

    # Submit tasks to the Dask scheduler
    futures = [
        client.submit(process_single_url, url_info)
        for url_info in url_info_list
    ]

    for future in futures:
        result_df = future.result()
        result_df = result_df.dropna()
        # result_df = result_df.groupby(['Latitude', 'Longitude']).mean().reset_index()

        for (start, end), file_name in zip(lon_ranges, file_names):
            filtered_df = result_df[(result_df['Longitude'] >= start) & (result_df['Longitude'] < end)]
            if not filtered_df.empty:
                filtered_df.to_csv(file_name, mode='a', header=False, index=False)
    
    print('Files saved into respective CSVs')
