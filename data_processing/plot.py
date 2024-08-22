import pandas as pd
import os
import sys

sys.path.append(os.path.abspath('.'))

from utils_dask import plot_polar_data

def choose_data(choice):
    
    if choice == 'Diviner-raw':
        home_dir = '../data/Diviner-temp/Diviner_CSVs'
        save_path = '../data/Diviner-temp/Diviner_temp.png'
        name = 'Diviner'
        title_prefix = 'Diviner temperature values'


    elif choice == 'Diviner-inter':
        home_dir = '../data/Diviner-temp/Diviner_interp_CSVs'
        save_path = '../data/Diviner-temp/Diviner_temp_interp.png'
        name = 'Diviner'
        title_prefix = 'Interpolated Diviner temperature values'


    elif choice == 'LOLA-raw':
        home_dir = '../data/LOLA-Albedo/LOLA_CSVs'
        save_path = '../data/LOLA-Albedo/LOLA_Albedo.png'
        name = 'LOLA'
        title_prefix = 'LOLA Albedo values'


    elif choice == 'LOLA-inter':
        home_dir = '../data/LOLA-Albedo/LOLA_interp_CSVs'
        save_path = '../data/LOLA-Albedo/LOLA_Albedo_interp.png'
        name = 'LOLA'
        title_prefix = 'Interpolated LOLA Albedo values'


    elif choice == 'M3-raw':
        home_dir = '../data/M3/M3_CSVs'
        save_path = '../data/M3/M3.png'
        name = 'M3'
        title_prefix = 'M3 CPR values'


    elif choice == 'M3-inter':
        home_dir = '../data/M3/M3_interp_CSVs'
        save_path = '../data/M3/M3_interp.png'
        name = 'M3'
        title_prefix = 'Interpolated M3 CPR values'


    elif choice == 'Mini-RF-raw':
        home_dir = '../data/Mini-RF/MiniRF_CSVs'
        save_path = '../data/Mini-RF/MiniRF_CPR.png'
        name = 'MiniRF'
        title_prefix = 'Mini-RF CPR values'

    elif choice == 'Mini-RF-inter':
        home_dir = '../data/Mini-RF/MiniRF_interp_CSVs'
        save_path = '../data/Mini-RF/MiniRF_CPR_interp.png'
        name = 'MiniRF'
        title_prefix = 'Interpolated Mini-RF CPR values'

    elif choice == 'M3-max':
        home_dir = '../data/M3/M3_CSVs_max'
        save_path = '../data/M3/M3_max.png'
        name = 'M3'
        title_prefix = 'M3 max values'

    elif choice == 'M3-elev':
        home_dir = '../data/M3/M3_CSVs'
        save_path = '../data/M3/M3_elev.png'
        name = 'Elevation'
        title_prefix = 'M3 elevation values'

    else:
        raise ValueError('Invalid choice')

    return home_dir, save_path, name, title_prefix


def main():
    # Options: 
    # 'Mini-RF-raw', 'Mini-RF-inter'
    # 'LOLA-raw', 'LOLA-inter'
    # 'Diviner-raw', 'Diviner-inter'
    # 'M3-raw', 'M3-inter'

    # home_dir, save_path, name, title_prefix = choose_data('M3-elev')
    name = 'Label'
    title_prefix = 'Combined data'
    save_path = '../data/Combined_CSVs/Combined_data.png'

    home_dir = '../data/Combined_CSVs'
    csvs = os.listdir(home_dir)
    print(csvs)
    dfs = [pd.read_csv(os.path.join(home_dir, csv)) for csv in csvs if csv.endswith('.csv')]
    # [print(df.describe(), '\n') for df in dfs]
    df = pd.concat(dfs, ignore_index=True)

    plot_polar_data(df, name, frac=0.5, title_prefix=title_prefix, save_path=save_path)


if __name__ == '__main__':
    main()