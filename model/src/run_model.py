import torch
import argparse
import pandas as pd
import numpy as np
import random
from models import FCNN, GCN
import joblib
import sys
from contextlib import contextmanager

from utils import get_random_filtered_graph, load_data


@contextmanager
def suppress_output():
    with open('/dev/null', 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def get_random_line(file_path, label_value=None, seed=42):
    with suppress_output():
        data = load_data(file_path, output=False)
    filtered_data = data if label_value is None else data[data['Label'] == label_value]
    return filtered_data.sample(n=1, random_state=seed).to_dict(orient='records')[0] if not filtered_data.empty else None


def main():
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_trials = 10

    #FCNN
    print("\nFCNN:")
    FCNN_model = FCNN(args.FCNN_input_dim, args.FCNN_hidden_dim, 1, args.FCNN_dropout_rate)
    FCNN_model.load_state_dict(torch.load(args.FCNN_load_path, map_location=device, weights_only=True))
    FCNN_model.to(device).eval()

    standardise_scalar_FCNN = joblib.load('../saved_models/standardise_scalar_FCNN.joblib')
    normalise_scalar_FCNN = joblib.load('../saved_models/normalise_scalar_FCNN.joblib')

    expected_columns = standardise_scalar_FCNN.feature_names_in_

    for i in range(num_trials):
        seed = 42 * i
        random.seed(seed)
        torch.manual_seed(seed)
        line = get_random_line(args.data_csv, i, seed) if i < 8 else get_random_line(args.data_csv, seed=seed)
        label = line['Label']

        data_df = pd.DataFrame([line]).drop(columns=['Label']).apply(pd.to_numeric)
        data_df = data_df.reindex(columns=expected_columns)

        data_np = standardise_scalar_FCNN.transform(data_df)
        data_df = pd.DataFrame(data_np, columns=expected_columns)
        data_np = normalise_scalar_FCNN.transform(data_df)
        
        data_tensor = torch.tensor(data_np, dtype=torch.float32).to(device)
        output = FCNN_model(data_tensor).squeeze().cpu().detach().numpy()
        lat = line['Latitude']
        lon = line['Longitude']
        print(f"Trial {i+1}, Lat: {lat:.2f}, Lon: {lon:.2f}, True label: {label}, Prediction: {output:.3f}")

    # GCN
    print("\nGCN:")
    GCN_model = GCN(args.GCN_input_dim, args.GCN_hidden_dim, 1, args.GCN_dropout_rate)
    GCN_model.load_state_dict(torch.load(args.GCN_load_path, map_location=device, weights_only=True))
    GCN_model.to(device).eval()

    standardise_scalar_GCN = joblib.load('../saved_models/standardise_scalar_GCN.joblib')
    normalise_scalar_GCN = joblib.load('../saved_models/normalise_scalar_GCN.joblib')

    k = 5
    # target_idx = k*k//2 + k//2

    for i in range(num_trials):
        seed = 42 * i
        random.seed(seed)
        torch.manual_seed(seed)
        with suppress_output():
            features, edge_index, labels, target_idx = get_random_filtered_graph(args.data_csv, i, k) if i < 8 else get_random_filtered_graph(args.data_csv, k=k)
        features_np = pd.DataFrame(features.cpu().numpy(), columns=['Latitude', 'Longitude', 'Diviner', 'LOLA', 'M3', 'MiniRF', 'Elevation'])
        lat_lon_df = features_np.loc[:, ["Latitude", "Longitude"]]

        features_np = features_np.drop(columns=["Latitude", "Longitude"])
        features_np = standardise_scalar_GCN.transform(features_np)
        features_np = normalise_scalar_GCN.transform(features_np)
        features_np = np.concatenate((lat_lon_df, features_np), axis=1)

        features = torch.tensor(features_np, dtype=torch.float32).to(device)
        features, edge_index = features.to(device), edge_index.to(device)
        output = GCN_model(features, edge_index).squeeze().cpu().detach().numpy()
        lat = lat_lon_df.iloc[target_idx]['Latitude']
        lon = lat_lon_df.iloc[target_idx]['Longitude']
        print(f"Trial {i+1}, Lat: {lat:.2f}, Lon: {lon:.2f}, True label: {labels[target_idx].item()}, Prediction: {output[target_idx]:.2f}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run points through the FCNN and GCN models.')
    parser.add_argument('--FCNN_input_dim', type=int, default=7, help='Input dimension of the FCNN model.')
    parser.add_argument('--FCNN_hidden_dim', type=int, default=512, help='Dimension of the hidden layer in the FCNN model.')
    parser.add_argument('--FCNN_dropout_rate', type=float, default=0.3, help='Dropout rate for the FCNN model.')
    parser.add_argument('--FCNN_load_path', type=str, default='../saved_models/FCNN.pth', help='Path to the saved FCNN model.')
    parser.add_argument('--GCN_input_dim', type=int, default=7, help='Input dimension of the GCN model.')
    parser.add_argument('--GCN_hidden_dim', type=int, default=512, help='Dimension of the hidden layer in the GCN model.')
    parser.add_argument('--GCN_dropout_rate', type=float, default=0.3, help='Dropout rate for the GCN model.')
    parser.add_argument('--GCN_load_path', type=str, default='../saved_models/GCN.pth', help='Path to the saved GCN model.')
    parser.add_argument('--data_csv', type=str, default='../../data/Combined_CSVs/', help='Path to the data file.')
    return parser.parse_args()

if __name__ == '__main__':
    main()