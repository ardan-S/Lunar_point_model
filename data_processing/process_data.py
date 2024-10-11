import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.utils.load_dfs import load_lro_df, load_m3_df
from data_processing.utils.interp import interpolate
from data_processing.utils.label import combine, label
from data_processing.utils.utils import load_dataset_config


def main(args):
    dataset_dict = load_dataset_config('../dataset_config.json', args)

    load_lro_df(dataset_dict['Diviner'], 'Diviner', debug=True)
    load_lro_df(dataset_dict['LOLA'], 'LOLA', debug=True)
    load_m3_df(dataset_dict['M3'], debug=True)
    load_lro_df(dataset_dict['MiniRF'], 'MiniRF', debug=True)
    sys.stdout.flush()

    # interpolate(dataset_dict['Diviner'], 'Diviner', plot_save_path=dataset_dict['Diviner']['plot_path'], debug=True)
    # interpolate(dataset_dict['LOLA'], 'LOLA', plot_save_path=dataset_dict['LOLA']['plot_path'], debug=True)
    # interpolate(dataset_dict['M3'], 'M3', plot_save_path=dataset_dict['M3']['plot_path'],  debug=True)
    interpolate(dataset_dict['MiniRF'], 'MiniRF', plot_save_path=dataset_dict['MiniRF']['plot_path'],  debug=True)
    print("Dataframes interpolated"); sys.stdout.flush()

    # combined_df = combine(diviner_df, lola_df, m3_df, mini_rf_df)
    # labeled_df = label(combined_df)
    # print("Dataframes combined and labeled")
    # print(f"Combined dataframe shape: {labeled_df.shape}")


def parse_args():
    parser = argparse.ArgumentParser(description="Process data")
    parser.add_argument("--download_dir", type=str, default="../../data/raw")
    parser.add_argument("--save_dir", type=str, default="../../data/CSVs/raw")
    parser.add_argument("--interp_dir", type=str, default="../../data/CSVs/interpolated")
    parser.add_argument("--plot_dir", type=str, default="../../data/plots")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
