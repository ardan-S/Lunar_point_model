import numpy as np
import sys
import os
from dask.distributed import Client
import asyncio
import argparse

sys.path.append(os.path.abspath('../../data_processing'))
from process_urls_dask import get_file_urls_async, process_urls_in_parallel
from utils_dask import plot_polar_data


def main(n_workers, threads_per_worker, memory_limit):
    # Set up the Dask client
    client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

    LOLA_home = 'https://imbrium.mit.edu/DATA/LOLA_GDR/POLAR/JP2/'

    async def process():
        # LOLA_urls = get_file_urls(LOLA_home, '.LBL', 'LDRM')    # LDRM is Lunar Digital Reflectance Map (Albedo)
        LOLA_urls = await get_file_urls_async(LOLA_home, '.LBL', 'LDRM')
        LOLA_df = process_urls_in_parallel(client, LOLA_urls, 'LOLA', num_processes=n_workers).compute()

        print(f'Number of NaNs in LOLA Albedo data: {np.isnan(LOLA_df["LOLA"]).sum()} out of {np.prod(LOLA_df["LOLA"].shape)} ({(np.isnan(LOLA_df["LOLA"]).sum()/np.prod(LOLA_df["LOLA"].shape)*100):.2f}%)\n')
        print(LOLA_df.describe())

        plot_polar_data(LOLA_df, 'LOLA', frac=None, title_prefix='LOLA Albedo', save_path='LOLA_Albedo.png')

    asyncio.run(process())
    print('Closing client...')
    client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LOLA script arguments')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--threads_per_worker', type=int, default=2, help='Number of threads per worker')
    parser.add_argument('--memory_limit', type=str, default='4GB', help='Memory limit per worker')
    args = parser.parse_args()
    main(args.n_workers, args.threads_per_worker, args.memory_limit)
