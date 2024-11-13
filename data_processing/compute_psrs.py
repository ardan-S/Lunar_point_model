import pandas as pd
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import sys

from utils.illum_conditions import load_and_prepare_data, compute_horizons_chunks
from utils.utils import plot_polar_data


def plot_elev_grid(df_n, df_s, sample=0.25, save_path=None):
    print(f"Plotting {len(df_n)} points for North Pole and {len(df_s)} points for South Pole"); sys.stdout.flush()
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(20, 10))

    assert 0 < sample <= 1, "Sample value should be between 0 and 1"
    assert len(df_n.index) > 0 or len(df_s.index) > 0, "No data to plot"
    assert np.isfinite(df_n['Longitude']).all() and np.isfinite(df_n['Latitude']).all() and np.isfinite(df_n['Elevation']).all(), "North Pole data contains nans"
    assert np.isfinite(df_s['Longitude']).all() and np.isfinite(df_s['Latitude']).all() and np.isfinite(df_s['Elevation']).all(), "South Pole data contains nans"

    df_n = df_n.sample(frac=sample)
    df_s = df_s.sample(frac=sample)

    def set_latitude_labels(ax, pole):
        ax.set_ylim(0, 15)
        ax.set_yticks(range(0, 16, 5))
        labels = [str(90 - x) if pole == 'north' else str(-90 + x) for x in range(0, 16, 5)]
        ax.set_yticklabels(labels)

    def plot_pole_data(ax, df, pole):
        if len(df.index) == 0:
            return
        sc = ax.scatter(df['theta'], df['r'], c=df['Elevation'], cmap='Greys_r', s=50)
        plt.colorbar(sc, ax=ax, label='Elevation')
        set_latitude_labels(ax, pole)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title(f'Elevation values - {pole.capitalize()} Pole')

    # Plot for North Pole
    if len(df_n.index) != 0:
        plot_pole_data(ax1, df_n, 'north')
    else:
        print('No data for North Pole')
        fig.delaxes(ax1)

    # Plot for South Pole
    if len(df_s.index) != 0:
        plot_pole_data(ax2, df_s, 'south')
    else:
        print('No data for South Pole')
        fig.delaxes(ax2)

    if save_path:
        plt.savefig(f"{save_path}/PSR_Elevation_plot.png")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def main(args):

    elev_grid, df_north, df_south = load_and_prepare_data(args)
    # plot_elev_grid(df_north, df_south, save_path=args.save_dir)

    df_north = df_north.head(1000)
    compute_horizons_chunks(df_north, 'North', args, n_splits = args.n_workers*3)
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
