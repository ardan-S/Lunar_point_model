import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd
import pandas as pd


# Function to split list into chunks of specified size
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def plot_polar_data(df, variable, frac=None, random_state=42, title_prefix='', save_path=None):

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

    ave_ddf = ddf.groupby(['Latitude', 'Longitude']).mean().reset_index().compute()
    north_pole_df = ave_ddf[ave_ddf['Latitude'] >= 0].copy()
    south_pole_df = ave_ddf[ave_ddf['Latitude'] < 0].copy()

    def prepare_polar_data(df, pole):
        if df.empty:
            return df
        df = df.copy()
        df['r'] = 90 - df['Latitude'] if (pole == 'north') else 90 + df['Latitude']
        df['theta'] = np.deg2rad(df['Longitude'])
        return df

    north_pole_df = prepare_polar_data(north_pole_df, 'north')
    south_pole_df = prepare_polar_data(south_pole_df, 'south')

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(20, 10))

    def set_latitude_labels(ax, pole):
        ax.set_ylim(0, 15)
        ax.set_yticks(range(0, 16, 5))
        labels = [str(90 - x) if pole == 'north' else str(-90 + x) for x in range(0, 16, 5)]
        ax.set_yticklabels(labels)

    def plot_pole_data(ax, df, pole):
        if df.empty:
            return
        sc = ax.scatter(df['theta'], df['r'], c=df[variable], cmap='Greys', s=50)
        plt.colorbar(sc, ax=ax, label=variable)
        set_latitude_labels(ax, pole)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title(f'{title_prefix} - {pole.capitalize()} Pole')

    # Plot for North Pole
    if not north_pole_df.empty:
        plot_pole_data(ax1, north_pole_df, 'north')
    else:
        print('No data for North Pole')
        fig.delaxes(ax1)

    # Plot for South Pole
    if not south_pole_df.empty:
        plot_pole_data(ax2, south_pole_df, 'south')
    else:
        print('No data for South Pole')
        fig.delaxes(ax2)

    print(f"Plotting {len(north_pole_df[variable]) + len(south_pole_df[variable])} data points...")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
