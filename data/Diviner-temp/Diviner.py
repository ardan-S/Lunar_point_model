import numpy as np
import sys
import os
from dask.distributed import Client
import asyncio
import argparse

sys.path.append(os.path.abspath('../data_processing'))
from process_urls_dask import get_file_urls_async, process_urls_in_parallel
# from utils_dask import plot_polar_data


def main(n_workers, threads_per_worker, memory_limit):
    print('Starting Diviner client...')
    client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

    temp_home = 'https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/data_derived_gdr_l3/2016/polar/jp2/'

    async def process():
        temp_urls = await get_file_urls_async(temp_home, '.lbl', 'tbol')
        temp_urls = temp_urls[:5]
        csv_path = './Diviner-temp/Diviner_CSVs'
        process_urls_in_parallel(client, temp_urls, 'Diviner', csv_path)

    # plot_polar_data(temp_df, 'Diviner', frac=None, title_prefix='Diviner temperature', save_path='Diviner_temp.png')

    asyncio.run(process())
    print('Closing Diviner client...')
    client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diviner script arguments')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--threads_per_worker', type=int, default=2, help='Number of threads per worker')
    parser.add_argument('--memory_limit', type=str, default='4GB', help='Memory limit per worker')
    args = parser.parse_args()
    main(args.n_workers, args.threads_per_worker, args.memory_limit)
