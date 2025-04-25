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
import requests


def clear_dir(home_dir, dirs_only=True):
    for file in os.listdir(home_dir):
        full_path = os.path.join(home_dir, file)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)    # Remove directories and their contents
        elif not dirs_only:
            os.remove(full_path)


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


# async def download_lola(download_dir, home_url, session, semaphore, keyword='lro', test=False):

#     os.makedirs(download_dir, exist_ok=True)

#     lasers = ["laser1", "laser2"]
#     tasks = []
#     n_tot = n_down = 0

#     for laser in lasers:
#         # Get subdirs for each laser
#         laser_url = urljoin(home_url, f"{laser}/")
#         try:
#             async with semaphore, session.get(laser_url) as response:
#                 response.raise_for_status()
#                 text = await response.text()
#         except Exception as e:
#             print(f"Failed to fetch LOLA data from {laser_url}. Error: {e}")
#             continue

#         soup = BeautifulSoup(text, 'html.parser')
#         subdirs = [
#             a['href'].rstrip('/') for a in soup.find_all('a', href=True)
#             if a['href'].endswith('/')
#         ]

#         # Download .tab files from each subdir
#         for subdir in subdirs:
#             subdir_url = urljoin(laser_url, f"{subdir}/")
#             try:
#                 async with semaphore, session.get(subdir_url) as response2:
#                     response2.raise_for_status()
#                     text2 = await response2.text()
#             except Exception as e:
#                 print(f"Failed to fetch LOLA data from {subdir_url}. Error: {e}")
#                 continue

#             soup2 = BeautifulSoup(text2, 'html.parser')
#             tab_files = [
#                 a['href'] for a in soup2.find_all('a', href=True)
#                 if a['href'].lower().endswith('.tab')
#             ]
#             n_tot += len(tab_files)

#             for tab in tab_files:
#                 file_url = urljoin(subdir_url, tab)
#                 tasks.append(download_file(session, file_url, download_dir, semaphore))
#                 n_down += 1
#                 if test:
#                     break
#             if test:
#                 break
        
#     await asyncio.gather(*tasks)
#     print(f"Downloaded {n_down} LOLA files out of {n_tot} files")


async def _fetch(session, url, semaphore):
    """Return the directory listing for *url* or None on failure."""
    try:
        async with semaphore, session.get(url) as r:
            r.raise_for_status()
            return await r.text()
    except Exception as exc:
        print(f"Failed to fetch {url}. Error: {exc}")
        return None


def _split_listing(html):
    """Return ([dirs], [tab_files]) from a directory listing page."""
    soup = BeautifulSoup(html, "html.parser")
    dirs = [a["href"].rstrip("/")
            for a in soup.find_all("a", href=True)
            if a["href"].endswith("/") and not a["href"].startswith("..")]
    tabs = [a["href"]
            for a in soup.find_all("a", href=True)
            if a["href"].lower().endswith(".tab")]
    return dirs, tabs


async def download_lola(download_dir, home_url, session, semaphore,
                        keyword="lro", test=False):

    os.makedirs(download_dir, exist_ok=True)

    lasers = ["laser1", "laser2"]
    tasks, n_tot, n_down = [], 0, 0

    for laser in lasers:
        laser_url = urljoin(home_url, f"{laser}/")
        listing = await _fetch(session, laser_url, semaphore)
        if listing is None:
            continue

        subdirs_lvl1, _ = _split_listing(listing)

        for sub1 in subdirs_lvl1:
            lvl1_url = urljoin(laser_url, f"{sub1}/")
            lvl1_html = await _fetch(session, lvl1_url, semaphore)
            if lvl1_html is None:
                continue

            # First try to harvest .tab files right here
            _, tab_files = _split_listing(lvl1_html)

            # If none found, the folder (e.g. polarpatch_np/sp) has another layer
            if not tab_files:
                subdirs_lvl2, _ = _split_listing(lvl1_html)

                for sub2 in subdirs_lvl2:              # one level deeper
                    lvl2_url = urljoin(lvl1_url, f"{sub2}/")
                    lvl2_html = await _fetch(session, lvl2_url, semaphore)
                    if lvl2_html is None:
                        continue

                    _, tab_files2 = _split_listing(lvl2_html)
                    n_tot += len(tab_files2)

                    for tab in tab_files2:
                        file_url = urljoin(lvl2_url, tab)
                        tasks.append(download_file(session, file_url,
                                                   download_dir, semaphore))
                        n_down += 1
                        if test:
                            break
                    if test:
                        break

            else:
                n_tot += len(tab_files)
                for tab in tab_files:
                    file_url = urljoin(lvl1_url, tab)
                    tasks.append(download_file(session, file_url,
                                               download_dir, semaphore))
                    n_down += 1
                    if test:
                        break

            if test:
                break

    await asyncio.gather(*tasks)
    print(f"Downloaded {n_down} LOLA files out of {n_tot} found")


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
            # 'url': 'https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/polar/jp2/',
            'url': 'https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_radr/',
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
    parser.add_argument("--existing_dirs", type=str, default="Skip", help="Replace or Skip existing directories")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
