"""
Diviner Data Processing Script

This script processes temperature data from the Diviner mission using Dask for parallel processing. 
It fetches URLs of data files, processes them in parallel, and stores the results in CSV format.

Usage:
    python Diviner.py --n_workers <number_of_workers> --threads_per_worker <threads_per_worker> --memory_limit <memory_limit>
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
    Initializes the Dask client and processes temperature data from the Diviner mission.

    Parameters:
    n_workers (int): Number of workers to use for parallel processing.
    threads_per_worker (int): Number of threads per worker.
    memory_limit (str): Memory limit per worker (e.g., '4GB').

    Returns:
    None
    """
    # Create a Dask client
    client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

    # Define the URL of the Diviner temperature data
    temp_home = 'https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/data_derived_gdr_l3/2016/polar/jp2/'

    async def process():
        # Get the URLs of the temperature data files
        temp_urls = await get_file_urls_async(temp_home, '.lbl', 'tbol')
        csv_path = './Diviner-temp/Diviner_CSVs'
        # Process the temperature data files in parallel
        process_urls_in_parallel(client, temp_urls, 'Diviner', csv_path)

    # Run the processing task
    asyncio.run(process())

    # Close the Dask client
    client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diviner script arguments')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--threads_per_worker', type=int, default=2, help='Number of threads per worker')
    parser.add_argument('--memory_limit', type=str, default='4GB', help='Memory limit per worker')
    args = parser.parse_args()
    main(args.n_workers, args.threads_per_worker, args.memory_limit)
