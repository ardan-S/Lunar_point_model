import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from multiprocessing import Pool
import pandas as pd

from src.process_image import download_parse_metadata, process_image


def get_file_urls(page_url, file_extension, keyword):
    response = requests.get(page_url)
    response.raise_for_status()  # Ensure we notice bad responses

    soup = BeautifulSoup(response.content, 'html.parser')
    urls = []

    # Find all anchor tags
    for anchor in soup.find_all('a'):
        href = anchor.get('href')
        if href and href.endswith(file_extension) and keyword in href:
            full_url = urljoin(page_url, href)
            urls.append(full_url)

    return urls


def process_urls_in_parallel(urls, output_csv_path, label_type, num_processes=4):
    url_info_list = [(url, output_csv_path, label_type) for url in urls]
    with Pool(num_processes) as pool:
        dfs = pool.map(process_url, url_info_list)
    return pd.concat(dfs, axis=0)


def process_url(url_info):
    label_url, output_csv_path, label_type = url_info
    if label_type == 'LOLA':
        image_url = label_url.replace('_JP2.LBL', '.JP2')
    else:
        image_url = label_url.replace('.lbl', '.jp2')
    metadata = download_parse_metadata(label_url)
    df = process_image(metadata, image_url, output_csv_path, label_type)
    return df
