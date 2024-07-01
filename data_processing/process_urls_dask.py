import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import dask.dataframe as dd
# from dask.distributed import Client

import aiohttp
# import asyncio

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
    return await process_image_async(metadata, image_url, label_type, output_csv_path)


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


def process_urls_in_parallel(client, urls, label_type, output_csv_path=None, num_processes=4):
    # Create a list of tuples containing the URL, output CSV path, and label type
    url_info_list = [(url, construct_image_url(url, label_type), output_csv_path, label_type) for url in urls]

    # Submit tasks to the Dask scheduler using client.submit
    futures = [
        # client.submit(process_image, client.submit(download_parse_metadata, url_info[0]), url_info[1], url_info[3], url_info[2])
        client.submit(process_single_url, url_info)
        for url_info in url_info_list
    ]

    # Gather the results from the futures and compute
    results = client.gather(futures)
    dfs = client.compute(results, sync=True)

    # Concatenate the DataFrames into a single Dask DataFrame and partition it
    combined_df = pd.concat(dfs, axis=0)
    return dd.from_pandas(combined_df, npartitions=num_processes)


# def process_url(url_info):
#     label_url, output_csv_path, label_type = url_info
#     image_url = construct_image_url(label_url, label_type)

#     try:
#         metadata = download_parse_metadata(label_url)
#         return process_image(metadata, image_url, label_type, output_csv_path)
#     except requests.HTTPError as e:
#         print(f'HTTP error occured: {e}' if e.response.status_code != 404 else f'URL not found: {label_url}')
#     return pd.DataFrame()  # Return an empty DataFrame in case of an error
