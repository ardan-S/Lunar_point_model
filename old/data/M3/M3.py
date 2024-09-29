"""
M3 data processing script

This script processes M3 data using Dask for parallel processing.
It fetche data files either as URLs or local directory, processes them in parallel, and stores the results in CSV format.

Usage:
    python M3.py --n_workers <number_of_workers> --threads_per_worker <threads_per_worker> --memory_limit <memory_limit>
"""


import sys
import os
from dask.distributed import Client
import asyncio
import argparse
import time
from dask import config as cfg
import signal

sys.path.append(os.path.abspath('../data_processing'))
from process_urls_dask import get_M3_urls_async, process_urls_in_parallel
from utils_dask import chunks

cfg.set({'distributed.scheduler.worker-ttl': '1h'})  # Set worker time to live to 1 hour
cfg.set({'distributed.worker.timeout': '1h'})  # Increase worker timeout
cfg.set({'distributed.scheduler.worker-saturation': 2})  # Increase worker saturation threshold

client = None


def handle_signal(signum, frame):
    """
    Gracefully handle SIGTERM and SIGINT signals by shutting down the Dask client and exiting the script.
    Used to ensure that the Dask client is properly closed when walltime limit is reached.
    """
    global client
    if client:
        client.shutdown()
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


def main(n_workers, threads_per_worker, memory_limit):
    """
    Initializes the Dask client and processes M3 data.

    Parameters:
    n_workers (int): Number of workers to use for parallel processing.
    threads_per_worker (int): Number of threads per worker.
    memory_limit (str): Memory limit per worker (e.g., '4GB').

    Returns:
    None
    """
    global client
    # Create a Dask client
    client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit, timeout='120s')

    # M3_home = 'https://planetarydata.jpl.nasa.gov/img/data/m3/CH1M3_0004_md5.txt'   # For retrieving from web
    M3_home = '/rds/general/user/as5023/ephemeral/as5023/M3/raw_files'        # For retrieving from local directory

    async def process():
        """
        Processes M3 data files in parallel. Fetches URLs of data files, processes them in parallel, and stores the results in CSV format.
        Given the number of files, the process is split into chunks of 20 files each for parallel processing.

        Parameters:
        None
        """
        # M3_urls = await get_M3_urls_async(M3_home, '.LBL', 'DATA')    # For retrieving from web
        M3_lbl_urls = [(os.path.join(M3_home, f)) for f in os.listdir(M3_home) if f.endswith('_L2.LBL')]    # For retrieving from local directory
        csv_path = '/rds/general/user/as5023/home/irp-as5023/data/M3/M3_CSVs'
        iter = 1
        start_time = time.time()

        for lbl_urls_in_chunk in chunks(M3_lbl_urls, 20):
            check_time = time.time()
            print(f'Processing chunk {iter}...')

            # Process the M3 data files in parallel
            process_urls_in_parallel(client, lbl_urls_in_chunk, 'M3', csv_path)
            print(f'Chunk {iter} completed in {(time.time() - check_time)/60:.2f} mins. Total time: {(time.time() - start_time)/60:.2f} mins.\n')
            iter += 1

    # Run the processing task
    asyncio.run(process())

    # Close the Dask client
    client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='M3 script arguments')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--threads_per_worker', type=int, default=2, help='Number of threads per worker')
    parser.add_argument('--memory_limit', type=str, default='4GB', help='Memory limit per worker')
    args = parser.parse_args()
    main(args.n_workers, args.threads_per_worker, args.memory_limit)
