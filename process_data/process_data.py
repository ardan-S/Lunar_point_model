import argparse

from process_data.utils.load_dfs import load_lro_df, load_m3_df
from process_data.utils.interp import interpolate
from process_data.utils.label import combine, label


def main(args):
    download_dir = args.download_dir

    dataset_dict = {
        'Diviner': {
            'file_path': f"{download_dir}/Diviner",
            'Orbiter': "LRO",
            'img_ext': '.jp2',
            'lbl_ext': '.lbl',
            'address': 'UNCOMPRESSED_FILE.IMAGE'
        },
        'LOLA': {
            'file_path': f"{download_dir}/LOLA",
            'Orbiter': "LRO",
            'img_ext': '.jp2',
            'lbl_ext': '_jp2.lbl',
            'address': 'UNCOMPRESSED_FILE.IMAGE'
        },
        'M3': {
            'file_path': f"{download_dir}/M3",
            'Orbiter': "Chandrayaan-1",
            'img_ext': '_RFL.IMG',
            'lbl_ext': '_L2.LBL',
            'loc_img_ext': '_LOC.IMG',
            'loc_lbl_ext': '_L1B.LBL',
            'address': 'RFL_FILE.RFL_IMAGE'
        },
        'MiniRF': {
            'file_path': f"{download_dir}/Mini-RF",
            'Orbiter': "LRO",
            'img_ext': '.img',
            'lbl_ext': '.lbl',
            'address': 'IMAGE'
        },
    }

    diviner_df = load_lro_df(dataset_dict['Diviner'], 'Diviner')
    lola_df = load_lro_df(dataset_dict['LOLA'], 'LOLA')
    m3_df = load_m3_df(dataset_dict['M3'])
    mini_rf_df = load_lro_df(dataset_dict['Mini-RF'], 'MiniRF')
    print("Dataframes created")

    diviner_df = interpolate(diviner_df)
    lola_df = interpolate(lola_df)
    m3_df = interpolate(m3_df)
    mini_rf_df = interpolate(mini_rf_df)

    combined_df = combine(diviner_df, lola_df, m3_df, mini_rf_df)

    labeled_df = label(combined_df)

    print("Dataframes combined and labeled")
    print(f"Combined dataframe shape: {labeled_df.shape}")


def parse_args():
    parser = argparse.ArgumentParser(description="Process data")
    parser.add_argument("--download_dir", type=str, default="data/raw")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
