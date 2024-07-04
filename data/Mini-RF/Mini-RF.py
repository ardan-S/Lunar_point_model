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
    print('Starting Mini-RF client...')
    client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

    MiniRF_home = 'https://pds-geosciences.wustl.edu/lro/lro-l-mrflro-5-global-mosaic-v1/lromrf_1001/data/128ppd/'

    async def process():
        MiniRF_urls = await get_file_urls_async(MiniRF_home, '.lbl', 'cpr')  # 'cpr' - circular polarisation ratio
        csv_path = './Mini-RF/MiniRF_CSVs'
        process_urls_in_parallel(client, MiniRF_urls, 'MiniRF', csv_path)

    # plot_polar_data(MiniRF_df, 'MiniRF', frac=1, title_prefix='Mini-RF CPR', save_path='MiniRF_CPR.png')

    asyncio.run(process())
    print('Closing client...')
    client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mini-RF script arguments')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--threads_per_worker', type=int, default=2, help='Number of threads per worker')
    parser.add_argument('--memory_limit', type=str, default='4GB', help='Memory limit per worker')
    args = parser.parse_args()
    main(args.n_workers, args.threads_per_worker, args.memory_limit)
