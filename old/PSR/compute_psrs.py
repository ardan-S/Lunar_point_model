import pandas as pd
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from pathlib import Path

from utils.PSRs.illum_conditions import load_and_prepare_data, compute_horizons_chunks
from utils.utils import plot_polar_data
from utils.PSRs.psr_elev import plot_elev_grid


def main(args):
    start_time = time.time()
    df_north, df_south = load_and_prepare_data(args)
    print(f"Found {sum(1 for item in os.listdir(args.parquet_save_dir) if item.endswith('.parquet'))} parquet files in {args.parquet_save_dir}")
    # plot_elev_grid(df_north, df_south, save_path=args.save_dir)

    print(f"Computing horizon chunks after: {time.time() - start_time:.2f} seconds")
    compute_horizons_chunks(args)
    print("Done")

    # pq_files = list(Path(args.parquet_save_dir).glob('*.parquet'))
    # assert pq_files, "No data found in the parquet directory"
    # pq_files = pq_files[:1]  # For testing
    # print("PROCESSING ONLY ONE FILE FOR TESTING")

    # for pq_file in pq_files:
    #     print(f"Processing {pq_file}")
    #     df = pd.read_parquet(pq_file)
    #     print(df.head())
    #     break

    # start_time = datetime(2020, 1, 1, 0, 0, 0)
    # time_steps = range(0, round(18.6 * 365 * 24), 6)    # Simulate over lunar precession cycle (18.6 years, 6-hour time steps)

    # for azimuth in np.arange(0, 359.5, 0.5):
    #     df[f'Azimuth_{azimuth}'] = np.nan

    # # Compute terrain horizon elevation
    # df = compute_horizon_elevation(df)


    # for time_step in time_steps:
    #     filtered_df = df[df['PSR'] == 1]    # Only consider points not yet marked as lit

    #     # Compute sun position and sun horizon elevation
    #     sun_azimuths, sun_elevations = compute_sun_position(
    #         filtered_df['Latitude'].values, filtered_df['Longitude'].values, time_step, start_time)


    #     # Filter points where sun is above terrain horizon
    #     mask = sun_elevations > horizon_elevations

    #     df.loc[filtered_df.index[mask], 'PSR'] = 0  # Mark points as lit



def parse_args():
    parser = argparse.ArgumentParser(description="Process data")
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--upload_dir", type=str, default="../../data/CSVs/combined")
    parser.add_argument("--save_dir", type=str, default="../../data/CSVs/PSRs")
    parser.add_argument("--grid_resolution", type=float, default=240, help="Resolution of the grid in meters")
    parser.add_argument("--parquet_save_dir", type=str, default="../../data/Parquet")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
