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


def clear_dir(home_dir):
    for file in os.listdir(home_dir):
        full_path = os.path.join(home_dir, file)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)


async def download_file(session, url, download_dir):
    local_filename = os.path.join(download_dir, os.path.basename(url))
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            with open(local_filename, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)
    except Exception as e:
        print(f"Error downloading {url}: {e}")


async def download_diviner(download_dir, home_url, session):
    years = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
    tasks = []

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
        ]

        # Download each .jp2 file and its corresponding .lbl file
        for jp2_url in jp2_urls:
            lbl_url = jp2_url.replace('.jp2', '.lbl')

            print(jp2_url)
            print(lbl_url)

            # tasks.append(download_file(session, jp2_url, download_dir))
            # tasks.append(download_file(session, lbl_url, download_dir))
            break

        break

    # await asyncio.gather(*tasks)


async def download_lola(download_dir, home_url, session):
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
    ]

    for jp2_url in jp2_urls:
        lbl_url = jp2_url.replace('.jp2', '_jp2.lbl')

        print(jp2_url)
        print(lbl_url)

        # tasks.append(download_file(session, jp2_url, download_dir))
        # tasks.append(download_file(session, lbl_url, download_dir))

        break

    # await asyncio.gather(*tasks)


async def download_m3(download_dir, home_url, img_extension, lbl_extension, session, keyword='DATA'):
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

        print(img_url)
        print(lbl_url)

        # tasks.append(download_file(session, img_url, download_dir))
        # tasks.append(download_file(session, lbl_url, download_dir))

        break

    # await asyncio.gather(*tasks)


async def download_mini_rf(download_dir, home_url, session):
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
        if link['href'].endswith('.img') and 'cpr' in link['href'].lower()  # Filter for .IMG files with 'cpr'
    ]

    for img_url in img_urls:
        lbl_url = img_url.replace('.img', '.lbl')

        print(img_url)
        print(lbl_url)

        # tasks.append(download_file(session, img_url, download_dir))
        # tasks.append(download_file(session, lbl_url, download_dir))
        break

    # await asyncio.gather(*tasks)


def count_files_in_directory(directory):
    return sum(1 for entry in os.scandir(directory) if entry.is_file())


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

    async with aiohttp.ClientSession() as session:

        # Loop through each dataset in dataset_dict and execute the download process
        for dataset, info in dataset_dict.items():
            url = info['url']
            download_path = info['path']
            download_func = info['download_func']

            # Skip existing directories if the user specifies 'Skip'
            if os.path.exists(download_path) and args.existing_dirs == "Skip":
                print(f"Skipping {dataset} download as directory already exists")
                continue

            # Otherwise, create the directory and download the dataset
            os.makedirs(download_path, exist_ok=True)
            print(f"Downloading {dataset} to {download_path}")

            try:
                if dataset == 'M3':
                    await download_func(download_path, url, '_RFL.IMG', '_L2.LBL', session)
                elif dataset == 'M3_loc':
                    await download_func(download_path, url, '_LOC.IMG', '_L1B.LBL', session)
                else:
                    await download_func(download_path, url, session)
                num_files = count_files_in_directory(download_path)
                print(f"Downloaded {num_files} {dataset} files from web to {download_path}\n")
            except Exception as e:
                print(f"Failed to download {dataset} from {url}. Error: {e}\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_dir", type=str, default="data/raw")
    parser.add_argument("--existing_dirs", type=str, default="Replace", help="Replace or Skip")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
