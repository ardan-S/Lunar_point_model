import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd
import pandas as pd


def chunks(lst, n):
    """
    Function to split list into chunks of specified size.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def plot_polar_data(df, variable, frac=None, random_state=42, title_prefix='', save_path=None):
    """
    Function to plot polar data (or a fraction of it) on a map of the Moon.
    """
    # Check for required columns
    required_columns = ['Latitude', 'Longitude', variable]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {', '.join(missing_cols)}")

    # Convert to Dask DataFrame and drop missing values
    ddf = dd.from_pandas(df.dropna(subset=required_columns), npartitions=4) if isinstance(df, pd.DataFrame) else df.dropna(subset=required_columns)
    if not isinstance(ddf, dd.DataFrame):
        raise ValueError("Input 'df' must be a pandas or Dask DataFrame")

    if frac:
        ddf = ddf.sample(frac=frac, random_state=random_state)

    ave_ddf = ddf.groupby(['Longitude', 'Latitude']).mean().reset_index().compute()
    north_pole_ddf = (ave_ddf[ave_ddf['Latitude'] >= 0]).copy()
    south_pole_ddf = (ave_ddf[ave_ddf['Latitude'] < 0]).copy()
    print("Dataframe read in and converted to Dask")

    def prepare_polar_data(ddf, pole):
        if len(ddf.index) == 0:
            return ddf
        ddf = ddf.copy()
        ddf['r'] = 90 - ddf['Latitude'] if (pole == 'north') else 90 + ddf['Latitude']
        ddf['theta'] = np.deg2rad(ddf['Longitude'])
        return ddf

    print("Preparing data for plotting...")
    north_pole_ddf = prepare_polar_data(north_pole_ddf, 'north')
    south_pole_ddf = prepare_polar_data(south_pole_ddf, 'south')

    # Convert to pandas DataFrame
    north_pole_df = north_pole_ddf.compute() if isinstance(north_pole_ddf, dd.DataFrame) else north_pole_ddf
    south_pole_df = south_pole_ddf.compute() if isinstance(south_pole_ddf, dd.DataFrame) else south_pole_ddf

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(20, 10))

    def set_latitude_labels(ax, pole):
        ax.set_ylim(0, 15)
        ax.set_yticks(range(0, 16, 5))
        labels = [str(90 - x) if pole == 'north' else str(-90 + x) for x in range(0, 16, 5)]
        ax.set_yticklabels(labels)

    def plot_pole_data(ax, df, pole):
        if len(df.index) == 0:
            return
        sc = ax.scatter(df['theta'], df['r'], c=df[variable], cmap='Greys_r', s=50)
        plt.colorbar(sc, ax=ax, label=variable)
        set_latitude_labels(ax, pole)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title(f'{title_prefix} - {pole.capitalize()} Pole')

    # Plot for North Pole
    if len(north_pole_df.index) != 0:
        plot_pole_data(ax1, north_pole_df, 'north')
    else:
        print('No data for North Pole')
        fig.delaxes(ax1)

    # Plot for South Pole
    if len(south_pole_df.index) != 0:
        plot_pole_data(ax2, south_pole_df, 'south')
    else:
        print('No data for South Pole')
        fig.delaxes(ax2)

    print(f"Plotting {len(north_pole_df[variable]) + len(south_pole_df[variable])} data points...")

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def generate_mesh(RESOLUTION=0.24):
    """
    Generate a mesh of points for the Moon's poles with a specified resolution.
    """
    MOON_RADIUS = 1737.4  # Radius of the Moon in kilometers
    # Resolution in kilometers (240 meters)

    # Convert resolution to degrees (approximate, depends on latitude)
    # 1 degree latitude is roughly MOON_RADIUS * pi / 180 km
    resolution_deg = (RESOLUTION / (MOON_RADIUS * np.pi / 180))

    # Latitude ranges for the two poles
    lat_ranges = [(75, 90), (-90, -75)]
    lon_slices = [(0, 30), (30, 60), (60, 90),
                  (90, 120), (120, 150), (150, 180),
                  (180, 210), (210, 240), (240, 270),
                  (270, 300), (300, 330), (330, 360)]
    # Generate grid points for both regions
    def generate_grid(lat_range, lon_range, resolution_deg):
        lats = np.arange(lat_range[0], lat_range[1] + resolution_deg, resolution_deg)
        lons = np.arange(lon_range[0], lon_range[1] + resolution_deg, resolution_deg)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        return lon_grid, lat_grid

    # Generate meshes for each longitude slice
    meshes = []
    for lon_range in lon_slices:
        lon_grid_north, lat_grid_north = generate_grid(lat_ranges[0], lon_range, resolution_deg)
        lon_grid_south, lat_grid_south = generate_grid(lat_ranges[1], lon_range, resolution_deg)
        lon_lat_grid_north = np.column_stack((lon_grid_north.ravel(), lat_grid_north.ravel()))
        lon_lat_grid_south = np.column_stack((lon_grid_south.ravel(), lat_grid_south.ravel()))
        meshes.append((lon_lat_grid_north, lon_lat_grid_south))

    # print number of points in each mesh
    for i, (lon_lat_grid_north, lon_lat_grid_south) in enumerate(meshes):
        print(f"Mesh {i + 1}: {len(lon_lat_grid_north):,} points per pole. Total: {2 * len(lon_lat_grid_north):,} points.")
    print(f'Total points: {sum(len(lon_lat_grid_north) + len(lon_lat_grid_south) for lon_lat_grid_north, lon_lat_grid_south in meshes):,}')

    return meshes
