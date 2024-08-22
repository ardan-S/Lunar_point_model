"""
LOLA processing script

This script processes LOLA albedo data using Dask for parallel processing.
It fetches URLs of data files, processes them in parallel, and stores the results in CSV format.

Usage:
    python LOLA.py --n_workers <number_of_workers> --threads_per_worker <threads_per_worker> --memory_limit <memory_limit>
"""


import sys
import os
from dask.distributed import Client
import asyncio
import argparse

sys.path.append(os.path.abspath('../data_processing'))
from process_urls_dask import get_file_urls_async, process_urls_in_parallel


def main(n_workers, threads_per_worker, memory_limit):
    """
    Initializes the Dask client and processes albedo data from the LOLA mission.

    Parameters:
    n_workers (int): Number of workers to use for parallel processing.
    threads_per_worker (int): Number of threads per worker.
    memory_limit (str): Memory limit per worker (e.g., '4GB').

    Returns:
    None
    """
    # Create a Dask client
    client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

    # Define the URL of the LOLA albedo data
    LOLA_home = 'https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/polar/jp2/'

    async def process():
        # Get the URLs of the albedo data files
        LOLA_urls = await get_file_urls_async(LOLA_home, '.lbl', 'ldam')
        csv_path = './LOLA-Albedo/LOLA_CSVs'
        # Process the albedo data files in parallel
        process_urls_in_parallel(client, LOLA_urls, 'LOLA', csv_path)

    # Run the processing task
    asyncio.run(process())

    # Close the Dask client
    client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LOLA script arguments')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--threads_per_worker', type=int, default=2, help='Number of threads per worker')
    parser.add_argument('--memory_limit', type=str, default='4GB', help='Memory limit per worker')
    args = parser.parse_args()
    main(args.n_workers, args.threads_per_worker, args.memory_limit)
