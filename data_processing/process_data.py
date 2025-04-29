import argparse
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.utils.load_dfs import load_lro_df, load_m3_df, load_lola_df
from data_processing.utils.interp2 import interpolate
from data_processing.utils.label import combine, label
from data_processing.utils.utils import load_dataset_config


def main(args):
    start_time = time.time()
    dataset_dict = load_dataset_config('../../data_processing/dataset_config.json', args)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.download_dir, exist_ok=True)

    # # Temporarily disable plot saving
    # dataset_dict['Diviner']['plot_path'] = None
    # dataset_dict['LOLA']['plot_path'] = None
    # dataset_dict['M3']['plot_path'] = None
    # dataset_dict['MiniRF']['plot_path'] = None
    # print("REMINDER: Plot saving is disabled. ")

    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        load_futures = [
            executor.submit(load_lro_df, dataset_dict['Diviner'], 'Diviner', plot_frac=0.95),
            executor.submit(load_lola_df, dataset_dict['LOLA'], 'LOLA', plot_frac=0.95),
            executor.submit(load_m3_df, dataset_dict['M3']),
            executor.submit(load_lro_df, dataset_dict['MiniRF'], 'MiniRF', plot_frac=0.95),
        ]

        for future in as_completed(load_futures):
            try:
                future.result()
            except Exception as e:
                raise e

    print(f"Loading stage complete after {(time.time() - start_time) /60:.2f} mins\n"); sys.stdout.flush()

    os.makedirs(args.interp_dir, exist_ok=True)

    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        interp_futures = [
            executor.submit(interpolate, dataset_dict['Diviner'], 'Diviner', plot_save_path=dataset_dict['Diviner']['plot_path']),
            executor.submit(interpolate, dataset_dict['LOLA'], 'LOLA', plot_save_path=dataset_dict['LOLA']['plot_path']),
            executor.submit(interpolate, dataset_dict['M3'], 'M3', plot_save_path=dataset_dict['M3']['plot_path']),
            executor.submit(interpolate, dataset_dict['MiniRF'], 'MiniRF', plot_save_path=dataset_dict['MiniRF']['plot_path'])
        ]

        for future in as_completed(interp_futures):
            try:
                future.result()
            except Exception as e:
                raise e

    print(f"Interpolation stage complete after {(time.time() - start_time) /60:.2f} mins\n"); sys.stdout.flush()

    label(combine(dataset_dict['Diviner']['interp_dir'],
                  dataset_dict['LOLA']['interp_dir'],
                  dataset_dict['M3']['interp_dir'],
                  dataset_dict['MiniRF']['interp_dir'], 
                  n_workers=args.n_workers
                  ),
            dataset_dict, args.plot_dir
            )

    print(f"Dataframes labeled after {(time.time() - start_time) /60:.2f} mins\n"); sys.stdout.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Process data")
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--download_dir", type=str, default="../../data/raw")
    parser.add_argument("--save_dir", type=str, default="../../data/CSVs/raw")
    parser.add_argument("--interp_dir", type=str, default="../../data/CSVs/interpolated")
    parser.add_argument("--combined_dir", type=str, default="../../data/CSVs/combined")
    parser.add_argument("--plot_dir", type=str, default="../../data/plots")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
