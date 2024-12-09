import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_elev_grid(df_n, df_s, sample=0.25, save_path=None):
    print(f"Plotting {len(df_n)} points for North Pole and {len(df_s)} points for South Pole"); sys.stdout.flush()
    # fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(20, 10))
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'projection': 'polar'})
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 2, 1, projection='polar')
    ax2 = fig.add_subplot(1, 2, 2, projection='polar')

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
