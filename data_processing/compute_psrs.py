import pandas as pd
import argparse
from datetime import datetime
import numpy as np

from utils.illum_conditions import load_and_prepare_data, process_in_chunks
from utils.utils import plot_polar_data


def elevation_grid_to_df(elevation_grid, min_x, min_y, grid_res):
    # Get the number of rows and columns in the grid
    n_rows, n_cols = elevation_grid.shape
    
    # Calculate the real x and y coordinates for each index
    x_coords = min_x + np.arange(n_cols) * grid_res
    y_coords = min_y + np.arange(n_rows) * grid_res

    # Expand x_coords and y_coords to match the elevation grid's shape
    x_column = np.tile(x_coords, n_rows)
    y_column = np.repeat(y_coords, n_cols)
    
    # Flatten elevation grid to create a single column of elevation values
    elevation_column = elevation_grid.ravel()
    
    # Create the DataFrame
    df = pd.DataFrame({
        'Latitude': x_column,
        'Longitude': y_column,
        'elevation_grid': elevation_column
    })
    
    return df

def main(args):

    elev_grid, min_x, min_y = load_and_prepare_data(args, sample=0.0001)

    # df = elevation_grid_to_df(elev_grid, min_x, min_y, 240)
    # print(f"Converted back to df shape: {df.shape}")
    # print(df.head())
    # plot_polar_data(df, 'elevation_grid', save_path='../../data/CSVs/PSRs')

    # process_in_chunks(elev_grid, args)

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
