import numpy as np
import sys
import os
from dask.distributed import Client
import asyncio
import argparse

sys.path.append(os.path.abspath('../../data_processing'))
from process_urls_dask import get_M3_urls_async, process_urls_in_parallel
from utils_dask import plot_polar_data


def main(n_workers, threads_per_worker, memory_limit):
    # Set up the Dask client
    client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

    M3_home = 'https://planetarydata.jpl.nasa.gov/img/data/m3/CH1M3_0004_md5.txt'

    async def process():
        print("!!!!!! FIX NAN HANDLING FOR M3 DATA !!!!!!")
        # M3_urls = get_M3_urls(M3_data_home, '.LBL', 'DATA')
        M3_urls = await get_M3_urls_async(M3_home, '.LBL', 'DATA')
        M3_urls = M3_urls[:2]
        M3_df = process_urls_in_parallel(client, M3_urls, 'M3', num_processes=n_workers).compute()

        print(f'Number of NaNs in M3 data: {np.isnan(M3_df["M3"]).sum()} out of {np.prod(M3_df["M3"].shape)} ({(np.isnan(M3_df["M3"]).sum()/np.prod(M3_df["M3"].shape)*100):.2f}%)\n')
        print(M3_df.describe())

        # plot_polar_data(M3_df, 'M3', frac=None, title_prefix='M3 values', save_path='M3_values.png')

    asyncio.run(process())
    print('Closing client...')
    client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='M3 script arguments')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--threads_per_worker', type=int, default=2, help='Number of threads per worker')
    parser.add_argument('--memory_limit', type=str, default='4GB', help='Memory limit per worker')
    args = parser.parse_args()
    main(args.n_workers, args.threads_per_worker, args.memory_limit)
