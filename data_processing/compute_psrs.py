import pandas as pd
import argparse
from datetime import datetime
import numpy as np

from utils.illum_conditions import compute_sun_position, compute_horizon_elevation
from utils.utils import load_csvs_parallel

def main(args):
    df = load_csvs_parallel(args.upload_dir, args.n_workers)
    df.filter(items=['Longitude', 'Latitude', 'Elevation'])
    df['PSR'] = 1  # Assume all points are permanently shadowed regions (PSRs)

    start_time = datetime(2020, 1, 1, 0, 0, 0)
    time_steps = range(0, round(18.6 * 365 * 24), 6)    # Simulate over lunar precession cycle (18.6 years, 6-hour time steps)

    for azimuth in np.arange(0, 359.5, 0.5):
        df[f'Azimuth_{azimuth}'] = np.nan

    # Compute terrain horizon elevation
    df = compute_horizon_elevation(df)


    for time_step in time_steps:
        filtered_df = df[df['PSR'] == 1]    # Only consider points not yet marked as lit

        # Compute sun position and sun horizon elevation
        sun_azimuths, sun_elevations = compute_sun_position(
            filtered_df['Latitude'].values, filtered_df['Longitude'].values, time_step, start_time)


        # Filter points where sun is above terrain horizon
        mask = sun_elevations > horizon_elevations

        df.loc[filtered_df.index[mask], 'PSR'] = 0  # Mark points as lit



def parse_args():
    parser = argparse.ArgumentParser(description="Process data")
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--upload_dir", type=str, default="../../data/CSVs/combined")
    parser.add_argument("--save_dir", type=str, default="../../data/CSVs/PSRs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
