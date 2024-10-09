import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.utils.load_dfs import load_lro_df, load_m3_df
from data_processing.utils.interp import interpolate
from data_processing.utils.label import combine, label


def main(args):

    dataset_dict = {
        'Diviner': {
            'file_path': f"{args.download_dir}/Diviner",
            'save_path': f"{args.save_dir}/Diviner",
            'interp_dir': f"{args.interp_dir}/Diviner",
            'plot_path': f"{args.plot_dir}",
            'Orbiter': "LRO",
            'img_ext': '.jp2',
            'lbl_ext': '.lbl',
            'address': 'UNCOMPRESSED_FILE.IMAGE',
            'max': 500,
            'min': 0
        },
        'LOLA': {
            'file_path': f"{args.download_dir}/LOLA",
            'save_path': f"{args.save_dir}/LOLA",
            'interp_dir': f"{args.interp_dir}/LOLA",
            'plot_path': f"{args.plot_dir}",
            'Orbiter': "LRO",
            'img_ext': '.jp2',
            'lbl_ext': '_jp2.lbl',
            'address': 'COMPRESSED_FILE',
            'max': 2.0,
            'min': 0.0
        },
        'M3': {
            'file_path': f"{args.download_dir}/M3",
            'save_path': f"{args.save_dir}/M3",
            'interp_dir': f"{args.interp_dir}/M3",
            'plot_path': f"{args.plot_dir}",
            'Orbiter': "Chandrayaan-1",
            'img_ext': '_RFL.IMG',
            'lbl_ext': '_L2.LBL',
            'loc_img_ext': '_LOC.IMG',
            'loc_lbl_ext': '_L1B.LBL',
            'address': 'RFL_FILE.RFL_IMAGE',
            'loc_address': 'LOC_FILE.LOC_IMAGE',
            'max': 1,
            'min': 0

        },
        'MiniRF': {
            'file_path': f"{args.download_dir}/Mini-RF",
            'save_path': f"{args.save_dir}/Mini-RF",
            'interp_dir': f"{args.interp_dir}/Mini-RF",
            'plot_path': f"{args.plot_dir}",
            'Orbiter': "LRO",
            'img_ext': '.img',
            'lbl_ext': '.lbl',
            'address': 'IMAGE',
            'max': 1.5,
            'min': 0
        },
    }

    load_lro_df(dataset_dict['Diviner'], 'Diviner', debug=True)
    load_lro_df(dataset_dict['LOLA'], 'LOLA', debug=True)
    load_m3_df(dataset_dict['M3'], debug=True)
    load_lro_df(dataset_dict['MiniRF'], 'MiniRF', debug=True)


    interpolate(dataset_dict['Diviner'], 'Diviner', debug=True)

    # lola_df = interpolate(lola_df)
    # m3_df = interpolate(m3_df)
    # mini_rf_df = interpolate(mini_rf_df)
    # print("Dataframes interpolated")

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
