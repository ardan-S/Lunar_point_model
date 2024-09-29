"""
Mini-RF data processing script

This script processes Mini-RF data using Dask for parallel processing.
It fetches data files either as URLs or local directory, processes them in parallel, and stores the results in CSV format.

Usage:
    python Mini-RF.py --n_workers <number_of_workers> --threads_per_worker <threads_per_worker> --memory_limit <memory_limit>
"""


import sys
import os
from dask.distributed import Client
import asyncio
import argparse
from dask import config as cfg
import signal

sys.path.append(os.path.abspath('../data_processing'))
from process_urls_dask import get_file_urls_async, process_urls_in_parallel

cfg.set({'distributed.scheduler.worker-ttl': '1h'})  # Set worker time to live to 1 hour
cfg.set({'distributed.worker.timeout': '1h'})  # Increase worker timeout
cfg.set({'distributed.scheduler.worker-saturation': 2})  # Increase worker saturation threshold

client = None


# Gracefully handle exits when walltime limit is reached
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
    Initializes the Dask client and processes Mini-RF data.

    Parameters:
    n_workers (int): Number of workers to use for parallel processing.
    threads_per_worker (int): Number of threads per worker.
    memory_limit (str): Memory limit per worker (e.g., '4GB').

    Returns:
    None
    """    
    global client
    # Create a Dask client
    client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

    # MiniRF_home = 'https://pds-geosciences.wustl.edu/lro/lro-l-mrflro-5-global-mosaic-v1/lromrf_1001/data/128ppd/'
    MiniRF_home = '/rds/general/user/as5023/ephemeral/as5023/Mini-RF/raw_files'        # For retrieving from local directory

    async def process():
        # Get the URLs of the Mini-RF data files
        # MiniRF_urls = await get_file_urls_async(MiniRF_home, '.lbl', 'cpr')  # 'cpr' - circular polarisation ratio
        MRF_lbl_urls = [(os.path.join(MiniRF_home, f)) for f in os.listdir(MiniRF_home) if f.endswith('.lbl')]
        csv_path = '/rds/general/user/as5023/home/irp-as5023/data/Mini-RF/MiniRF_CSVs'

        # Process the Mini-RF data files in parallel
        process_urls_in_parallel(client, MRF_lbl_urls, 'MiniRF', csv_path)

    # Run the processing task
    asyncio.run(process())

    # Close the Dask client
    client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mini-RF script arguments')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--threads_per_worker', type=int, default=2, help='Number of threads per worker')
    parser.add_argument('--memory_limit', type=str, default='4GB', help='Memory limit per worker')
    args = parser.parse_args()
    main(args.n_workers, args.threads_per_worker, args.memory_limit)
