import numpy as np
import pandas as pd

import os
import sys

sys.path.append(os.path.abspath('.'))

from utils_dask import plot_polar_data


def main():
    Div_csvs = os.listdir('./Diviner-temp/Diviner_CSVs')

    # Diviner_df = pd.concat([pd.read_csv(f'./Diviner-temp/Diviner_CSVs/{csv}') for csv in Div_csvs])
    Diviner_df = pd.read_csv(f'./Diviner-temp/Diviner_CSVs/{Div_csvs[0]}')

    plot_polar_data(Diviner_df, 'Diviner', frac=0.25, title_prefix='Diviner temperature (Lon: 0 to 60)', save_path='Diviner_temp.png')


if __name__ == '__main__':
    main()