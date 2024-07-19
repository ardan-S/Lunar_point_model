import numpy as np
import sys
import os
import asyncio
import argparse
from dask import config as cfg
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
import subprocess
import time

sys.path.append(os.path.abspath('../data_processing'))
from process_urls_dask import get_M3_urls_async, construct_image_url

cfg.set({'distributed.scheduler.worker-ttl': '1h'})  # Set worker time to live to 1 hour
cfg.set({'distributed.worker.timeout': '1h'})  # Increase worker timeout
cfg.set({'distributed.scheduler.worker-saturation': 2})  # Increase worker saturation threshold

def download_file(url, download_dir):
    result = subprocess.run(
        ["wget", url, "-P", download_dir],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    if result.returncode != 0:
        raise ValueError(f"Failed to download {url}")

def main(n_workers):
    start_time = time.time()
    download_dir = Path(os.getenv('RDS')) / 'ephemeral' / 'as5023' / 'M3' / 'raw_files'
    # download_dir = Path(os.getenv('RDS')) / 'ephemeral' / 'as5023' / 'Mini-RF' / 'raw_files'

    # Download image and metadata files
    M3_home = 'https://planetarydata.jpl.nasa.gov/img/data/m3/CH1M3_0004_md5.txt'
    M3_calib= 'https://planetarydata.jpl.nasa.gov/img/data/m3/CH1M3_0003_md5.txt'
    M3_base = 'https://planetarydata.jpl.nasa.gov/img/data/m3/'

    # Get urls asynchronously
    async def process():
        M3_lbl_urls = await get_M3_urls_async(M3_home, '.LBL', 'DATA')
        M3_img_urls = [construct_image_url(url, 'M3') for url in M3_lbl_urls]

        M3_loc_lbl_urls = []
        M3_loc_img_urls = []
        response = requests.get(M3_calib)
        lines = response.text.splitlines()
        for line in lines:
            parts = line.split()
            if len(parts) == 2 and 'L1B/' in parts[1] and parts[1].endswith('L1B.LBL'):
                M3_loc_lbl_urls.append(os.path.join(M3_base, parts[1]))
            if len(parts) == 2 and 'L1B/' in parts[1] and parts[1].endswith('LOC.IMG'):
                M3_loc_img_urls.append(os.path.join(M3_base, parts[1]))

        print(f"Number of LBL files: {len(M3_lbl_urls)}, Number of IMG files: {len(M3_img_urls)}")
        print(f"Number of LOC LBL files: {len(M3_loc_lbl_urls)}, Number of LOC IMG files: {len(M3_loc_img_urls)}")
        sys.stdout.flush()
        print("\nFirst 3 of each group:")
        print(f"M3_lbl_urls: {M3_lbl_urls[:3]}")
        print(f"M3_img_urls: {M3_img_urls[:3]}")
        print(f"M3_loc_lbl_urls: {M3_loc_lbl_urls[:3]}")
        print(f"M3_loc_img_urls: {M3_loc_img_urls[:3]}\n")
        sys.stdout.flush()

        return np.concatenate([M3_lbl_urls, M3_img_urls, M3_loc_lbl_urls, M3_loc_img_urls])

    # async def process():
    #     MRF_lbl_url = 'https://pds-geosciences.wustl.edu/lro/lro-l-mrflro-5-global-mosaic-v1/lromrf_1001/data/128ppd/global_cpr_128ppd_simp_0c.lbl'
    #     MRF_img_url = 'https://pds-geosciences.wustl.edu/lro/lro-l-mrflro-5-global-mosaic-v1/lromrf_1001/data/128ppd/global_cpr_128ppd_simp_0c.img'
    #     return np.concatenate([[MRF_lbl_url, MRF_img_url]])

    urls = asyncio.run(process())
    print(f"Download directory: {download_dir}")
    os.makedirs(download_dir, exist_ok=True)

    # Download files in parallel
    print(f"Begin downloading {len(urls)} in parallel")
    sys.stdout.flush()
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(download_file, url, download_dir) for url in urls]
        for future in as_completed(futures):
            future.result()

    print(f"Download complete after {(time.time() - start_time)/60:.2f} mins")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='M3 script arguments')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers')
    args = parser.parse_args()
    main(args.n_workers)
