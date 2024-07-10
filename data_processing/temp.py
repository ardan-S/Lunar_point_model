import numpy as np
import pandas as pd

import os
import sys

sys.path.append(os.path.abspath('.'))

from utils_dask import plot_polar_data


def main():
    Div_csvs = os.listdir('../data/Diviner-temp/Diviner_CSVs')
    print(Div_csvs)
    # Diviner_df = pd.concat([pd.read_csv(f'../data/Diviner-temp/Diviner_CSVs/{csv}') for csv in Div_csvs])
    # Diviner_df = pd.read_csv(f'../data/Diviner-temp/Diviner_CSVs/{Div_csvs[0]}')
    save_path = '../data/Diviner-temp/Diviner_temp.png'
    print('Hello all')

    # plot_polar_data(Diviner_df, 'Diviner', frac=0.25, title_prefix='Diviner temperature (Lon: 0 to 60)', save_path=save_path)


if __name__ == '__main__':
    main()