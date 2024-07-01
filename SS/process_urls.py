import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from multiprocessing import Pool
import pandas as pd

from SS.process_image import download_parse_metadata, process_image


def get_file_urls(page_url, file_extension, keyword, limit=None):
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
            if limit and len(urls) >= limit:
                break

    return urls


def get_M3_urls(page_url, file_extension, keyword):

    response = requests.get(page_url)
    response.raise_for_status()  # Ensure we notice bad responses

    base_url = page_url.rsplit('/', 1)[0] + '/'
    urls = []

    # Process each line in the response text
    for line in response.text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            _, file_path = parts
            if file_path.endswith(file_extension) and keyword in file_path:
                full_url = urljoin(base_url, file_path)
                urls.append(full_url)

    return urls


def process_urls_in_parallel(urls, label_type, output_csv_path=None, num_processes=4):
    url_info_list = [(url, output_csv_path, label_type) for url in urls]
    with Pool(num_processes) as pool:
        dfs = pool.map(process_url, url_info_list)
        print(f"Processed {len(dfs)} images")
    return pd.concat(dfs, axis=0)


def process_url(url_info):
    label_url, output_csv_path, label_type = url_info

    if label_type == 'LOLA':
        image_url = label_url.replace('_JP2.LBL', '.JP2')
    elif label_type == 'M3-data':
        image_url = label_url.replace('_L2.LBL', '_RFL.IMG')
    elif label_type == 'M3-loc':
        image_url = label_url.replace('.HDR', '.IMG')
    elif label_type == 'MiniRF':
        image_url = label_url.replace('.lbl', '.img')
    else:
        image_url = label_url.replace('.lbl', '.jp2')   # Default to Diviner

    try:
        metadata = download_parse_metadata(label_url)
        # print(f'metadata: {metadata}')
        df = process_image(metadata, image_url, label_type, output_csv_path)
        return df
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            print(f"URL not found: {image_url}")
        else:
            print(f"HTTP error occurred: {e}")
    return pd.DataFrame()  # Return an empty DataFrame in case of an error
