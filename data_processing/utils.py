import numpy as np
import matplotlib.pyplot as plt


def plot_polar_data(df, variable, frac=None, random_state=42, title_prefix='', save_path=None):

    # Check for required columns
    required_columns = ['Latitude', 'Longitude', variable]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the DataFrame")

        df = df.dropna(subset=required_columns)

    sampled_df = df.sample(frac=frac, random_state=random_state) if frac else df
    ave_df = sampled_df.groupby(['Latitude', 'Longitude']).mean().reset_index()

    north_pole_df = ave_df[ave_df['Latitude'] >= 0].copy()
    south_pole_df = ave_df[ave_df['Latitude'] < 0].copy()

    def prepare_polar_data(df, pole):
        if df.empty:
            return df
        df = df.copy()
        min_lat = df['Latitude'].min()
        max_lat = df['Latitude'].max()
        print(f'Latitude range: {min_lat} to {max_lat}')
        print(f'Second latitude: {df["Latitude"].iloc[1]}')
        # df['r'] = (max_lat - df['Latitude']) / (max_lat - min_lat)
        df['r'] = (90 - df['Latitude']) if pole == 'north' else (90 + df['Latitude'])
        df['theta'] = np.deg2rad(df['Longitude'])
        return df

    north_pole_df = prepare_polar_data(north_pole_df, 'north')
    south_pole_df = prepare_polar_data(south_pole_df, 'south')

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(20, 10))

    def set_latitude_labels(ax, pole):
        if pole == 'north':
            ax.set_ylim(0, 15)
            ax.set_yticks(range(0, 16, 5))
            ax.set_yticklabels([str(90 - x) for x in range(0, 16, 5)])
        else:
            ax.set_ylim(0, 15)
            ax.set_yticks(range(0, 16, 5))
            ax.set_yticklabels([str(-90 + x) for x in range(0, 16, 5)])

    # Plot for North Pole
    if not north_pole_df.empty:
        sc1 = ax1.scatter(north_pole_df['theta'], north_pole_df['r'], c=north_pole_df[variable], cmap='Greys', s=50)
        plt.colorbar(sc1, ax=ax1, label=variable)
        # ax1.set_ylim(0, 1)
        # ax1.set_ylim(75, 90)
        set_latitude_labels(ax1, 'north')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_title(f'{title_prefix} - North Pole')
    else:
        print('No data for North Pole')
        fig.delaxes(ax1)

    # Plot for South Pole
    if not south_pole_df.empty:
        sc2 = ax2.scatter(south_pole_df['theta'], south_pole_df['r'], c=south_pole_df[variable], cmap='Greys', s=50)
        plt.colorbar(sc2, ax=ax2, label=variable)
        # ax2.set_ylim(0, 1)
        # ax2.set_ylim(-75, -90)
        set_latitude_labels(ax2, 'south')
        ax2.set_theta_zero_location('N')
        ax2.set_theta_direction(-1)
        ax2.set_title(f'{title_prefix} - South Pole')
    else:
        print('No data for South Pole')
        fig.delaxes(ax2)

    print(f"Plotted {len(north_pole_df[variable]) + len(south_pole_df[variable])} data points")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
