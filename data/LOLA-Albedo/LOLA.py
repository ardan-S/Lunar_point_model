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
    print('Starting LOLA client...')
    client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

    # LOLA_home = 'https://imbrium.mit.edu/DATA/LOLA_GDR/POLAR/JP2/'
    LOLA_home = 'https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/polar/jp2/'

    async def process():
        LOLA_urls = await get_file_urls_async(LOLA_home, '.lbl', 'ldam')
        csv_path = './LOLA-Albedo/LOLA_CSVs'
        process_urls_in_parallel(client, LOLA_urls, 'LOLA', csv_path)

    asyncio.run(process())
    print('Closing LOLA client...')
    client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LOLA script arguments')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--threads_per_worker', type=int, default=2, help='Number of threads per worker')
    parser.add_argument('--memory_limit', type=str, default='4GB', help='Memory limit per worker')
    args = parser.parse_args()
    main(args.n_workers, args.threads_per_worker, args.memory_limit)
