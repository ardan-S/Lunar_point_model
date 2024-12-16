"""
Files which need to be downloaded:
    Diviner: image file, label file
    Lola: image file, label file
    M3: image file, label file, loc image file, loc label file
    Mini-RF: image file, label file
"""

import argparse
import os
import shutil
from urllib.parse import urljoin
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import time
from pathlib import Path


def clear_dir(home_dir):
    for file in os.listdir(home_dir):
        full_path = os.path.join(home_dir, file)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)    # Remove directories and their contents


async def download_file(session, url, download_dir, semaphore):
    async with semaphore:
        local_filename = os.path.join(download_dir, os.path.basename(url))
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=14400)) as response:
                response.raise_for_status()
                with open(local_filename, 'wb') as f:
                    async for chunk in response.content.iter_chunked(4096):
                        f.write(chunk)
        except asyncio.TimeoutError:
            print(f"Timeout downloading {url}")
        except Exception as e:
            print(f"Error downloading {url}: {e} ({type(e).__name__})")


async def download_diviner(download_dir, home_url, session, semaphore, keyword='tbol', test=False):
    years = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
    tasks = []
    n_files_tot = 0
    n_files_down = 0

    for year in years:
        url = urljoin(home_url, f'{year}/polar/jp2/')

        try:
            async with session.get(url) as response:
                response.raise_for_status()
                content = await response.read()
        except Exception as e:
            print(f"Failed to fetch Diviner data from {url}. Error: {e}")
            continue

        soup = BeautifulSoup(content, 'html.parser')
        jp2_urls = [
            urljoin(url, link['href'])
            for link in soup.find_all('a', href=True)
            if link['href'].endswith('.jp2')
            and keyword in link['href']
        ]
        n_files_tot += len(jp2_urls)

        # Download each .jp2 file and its corresponding .lbl file
        for jp2_url in jp2_urls:
            lbl_url = jp2_url.replace('.jp2', '.lbl')
            tasks.append(download_file(session, jp2_url, download_dir, semaphore))
            tasks.append(download_file(session, lbl_url, download_dir, semaphore))
            n_files_down += 1
            if test:
                break


    await asyncio.gather(*tasks)
    if test:
        print(f"Test: Downloaded {n_files_down} Diviner file out of {n_files_tot} files")
    else:
        print(f"Downloaded {n_files_down} Diviner files")


async def download_lola(download_dir, home_url, session, semaphore, keyword='ldam', test=False):
    tasks = []
    try:
        async with session.get(home_url) as response:
            response.raise_for_status()
            content = await response.read()
    except Exception as e:
        print(f"Failed to fetch LOLA data from {home_url}. Error: {e}")
        return

    soup = BeautifulSoup(content, 'html.parser')
    jp2_urls = [
        urljoin(home_url, link['href'])  # Construct the full URL
        for link in soup.find_all('a', href=True)  # Find all <a> tags with href attribute
        if link['href'].endswith('.jp2')  # Filter for .jp2 files
        and keyword in link['href']  # Ensure the keyword (e.g., 'ldam') is in the file path
    ]

    for jp2_url in jp2_urls:
        lbl_url = jp2_url.replace('.jp2', '_jp2.lbl')
        tasks.append(download_file(session, jp2_url, download_dir, semaphore))
        tasks.append(download_file(session, lbl_url, download_dir, semaphore))
        if test:
            break

    await asyncio.gather(*tasks)
    if test:
        print(f"Test: Downloaded 1 LOLA file out of {len(jp2_urls)} files")
    else:
        print(f"Downloaded {len(jp2_urls)} LOLA files")


async def download_m3(download_dir, home_url, img_extension, lbl_extension, session, semaphore, keyword='DATA', test=False):
    base_url = home_url.rsplit('/', 1)[0] + '/'
    tasks = []
    try:
        async with session.get(home_url) as response:
            response.raise_for_status()
            text = await response.text()
    except Exception as e:
        print(f"Failed to fetch M3 data from {home_url}. Error: {e}")
        return

    img_urls = [
        urljoin(base_url, parts[1])  # Join the base URL with the file path
        for line in text.splitlines()  # Iterate through each line of text
        if (parts := line.split())  # Split each line by spaces
        and len(parts) == 2  # Ensure there are exactly two parts: checksum and file path
        and parts[1].endswith(img_extension)  # Filter by file extension
        and keyword in parts[1]  # Ensure the keyword (e.g., 'DATA') is in the file path
    ]

    for img_url in img_urls:
        lbl_url = img_url.replace(img_extension, lbl_extension)
        tasks.append(download_file(session, img_url, download_dir, semaphore))
        tasks.append(download_file(session, lbl_url, download_dir, semaphore))
        if test:
            break

    await asyncio.gather(*tasks)
    if test: 
        print(f"Test: Downloaded 1 M3 file out of {len(img_urls)} files")
    else:
        print(f"Downloaded {len(img_urls)} M3 files")


async def download_mini_rf(download_dir, home_url, session, semaphore, keyword='cpr', test=False):
    start_time = time.time()
    tasks = []
    try:
        async with session.get(home_url) as response:
            response.raise_for_status()
            text = await response.text()
    except Exception as e:
        print(f"Failed to fetch Mini-RF data from {home_url}. Error: {e}")
        return

    soup = BeautifulSoup(text, 'html.parser')
    img_urls = [
        urljoin(home_url, link['href'])  # Construct the full URL
        for link in soup.find_all('a', href=True)  # Find all <a> tags with href attribute
        if link['href'].endswith('.img')
        and keyword in link['href'].lower()  # Filter for .IMG files with 'cpr'
    ]

    for img_url in img_urls:
        lbl_url = img_url.replace('.img', '.lbl')
        tasks.append(download_file(session, img_url, download_dir, semaphore))
        tasks.append(download_file(session, lbl_url, download_dir, semaphore))
        if test:
            break

    await asyncio.gather(*tasks)
    if test:
        print(f"Test: Downloaded 1 Mini-RF file out of {len(img_urls)} files")
    else:
        print(f"Downloaded {len(img_urls)} Mini-RF files")


async def main(args):
    dataset_dict = {
        'Diviner': {
            'url': 'https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/data_derived_gdr_l3/',
            'path': os.path.join(args.download_dir, 'Diviner'),
            'download_func': download_diviner
        },
        'LOLA': {
            'url': 'https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/polar/jp2/',
            'path': os.path.join(args.download_dir, 'LOLA'),
            'download_func': download_lola
        },
        'M3': {
            'url': 'https://planetarydata.jpl.nasa.gov/img/data/m3/CH1M3_0004_md5.txt',
            'path': os.path.join(args.download_dir, 'M3'),
            'download_func': download_m3
        },
        'M3_loc': {
            'url': 'https://planetarydata.jpl.nasa.gov/img/data/m3/CH1M3_0003_md5.txt',
            'path': os.path.join(args.download_dir, 'M3'),
            'download_func': download_m3
        },
        'Mini-RF': {
            'url': 'https://pds-geosciences.wustl.edu/lro/lro-l-mrflro-5-global-mosaic-v1/lromrf_1001/data/128ppd/',
            'path': os.path.join(args.download_dir, 'Mini-RF'),
            'download_func': download_mini_rf
        }
    }

    os.makedirs(args.download_dir, exist_ok=True)   # Create download directory if it doesn't exist
    if args.existing_dirs == "Replace":             # Clear the download directory if the user specifies 'Replace'
        clear_dir(args.download_dir)

    semaphore = asyncio.Semaphore(10)   # Limit the number of concurrent downloads to 10 to prevent overloading the server

    async with aiohttp.ClientSession() as session:
        tasks = []
        for dataset, info in dataset_dict.items():  # Loop through each dataset in dataset_dict and execute the download process
            url = info['url']
            download_path = info['path']
            download_func = info['download_func']

            # Skip existing dataset directories if the user specifies 'Skip'
            if os.path.exists(download_path) and args.existing_dirs == "Skip":
                print(f"Skipping {dataset} download as directory already exists")
                continue

            # Otherwise, create the dataset directory (inside the download directory) and download the dataset
            os.makedirs(download_path, exist_ok=True)
            print(f"Downloading {dataset} to {download_path}")

            if dataset == 'M3':
                task = download_func(download_path, url, '_RFL.IMG', '_L2.LBL', session, semaphore, test=False)
            elif dataset == 'M3_loc':
                task = download_func(download_path, url, '_LOC.IMG', '_L1B.LBL', session, semaphore, test=False)
            else:
                task = download_func(download_path, url, session, semaphore, test=False)
            tasks.append(task)

        await asyncio.gather(*tasks)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_dir", type=str, default="../../data/raw", help="Directory to download data to")
    parser.add_argument("--existing_dirs", type=str, default="Replace", help="Replace or Skip existing directories")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
