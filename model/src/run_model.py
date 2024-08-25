import torch
import argparse
import pandas as pd
import numpy as np
import csv
import random
from models import FCNN, GCN
from sklearn.neighbors import NearestNeighbors


def get_random_filtered_graph(file_path, label_value=None, k=5):
    data = pd.read_csv(file_path)

    selected_row = data[data['Label'] == label_value].sample(n=1) if label_value else data.sample(n=1)
    if selected_row.empty:
        raise ValueError(f"No data found for label value {label_value}")
    
    selected_coords = selected_row[['Latitude', 'Longitude']].values
    nbrs = NearestNeighbors(n_neighbors=k*k, algorithm='ball_tree').fit(data[['Latitude', 'Longitude']].values)
    subgraph_data = data.iloc[nbrs.kneighbors(selected_coords)[1][0]]   # Get the indices of the k*k nearest neighbours

    node_features = torch.tensor(subgraph_data[['Latitude', 'Longitude', 'Diviner', 'LOLA', 'M3', 'MiniRF', 'Elevation']].values, dtype=torch.float32)
    node_labels = torch.tensor(subgraph_data['Label'].values, dtype=torch.float32)

    edge_index_list = []

    for i in range(k*k):
        for j in range(k*k):
            if i != j:
                edge_index_list.append([i, j])

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    self_loops = torch.arange(k*k, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, self_loops], dim=1)

    return node_features, edge_index, node_labels


def get_random_filtered_line(file_path, label_value):
    with open(file_path, 'r') as file:
        matching_lines = [row for row in csv.DictReader(file) if row['Label'] == str(label_value)]
    return random.choice(matching_lines) if matching_lines else None


def main():
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #FCNN
    FCNN_model = FCNN(args.FCNN_input_dim, args.FCNN_hidden_dim, 1, args.FCNN_dropout_rate)
    FCNN_model.load_state_dict(torch.load(args.FCNN_load_path, map_location=device))
    FCNN_model.to(device).eval()

    good_data_FCNN = get_random_filtered_line(args.data_csv, 7)
    mid_data_FCNN = get_random_filtered_line(args.data_csv, 3)
    bad_data_FCNN = get_random_filtered_line(args.data_csv, 0)

    print(f"Good data (FCNN): {good_data_FCNN}")
    print(f"Mid data (FCNN): {mid_data_FCNN}")
    print(f"Bad data (FCNN): {bad_data_FCNN}\n")

    # Convert to DataFrame and then to tensor
    for label, data in zip(['Good', 'Mid', 'Bad'], [good_data_FCNN, mid_data_FCNN, bad_data_FCNN]):
        data_df = pd.DataFrame([data]).drop(columns=['Label']).apply(pd.to_numeric)
        data_tensor = torch.tensor(data_df.values, dtype=torch.float32).to(device)
        output = FCNN_model(data_tensor).squeeze()
    
        print(f"{label} data tensor (FCNN): \n{data_tensor}\n")
        print(f"FCNN model output for {label} data: {output.cpu().detach().numpy()}\n")

    print("\n")

    # GCN
    GCN_model = GCN(args.GCN_input_dim, args.GCN_hidden_dim, 1, args.GCN_dropout_rate)
    GCN_model.load_state_dict(torch.load(args.GCN_load_path, map_location=device))
    GCN_model.to(device).eval()

    k = 5
    target_idx = k*k//2 +1

    for label_value, label in zip([7, 0], ['Good', 'Bad']):
        features, edge_index, labels = get_random_filtered_graph(args.data_csv, label_value, k)
        features, edge_index = features.to(device), edge_index.to(device)
        output = GCN_model(features, edge_index).squeeze()
        
        print(f"{label} label at centre node (GCN): {labels[target_idx]}")
        print(f"GCN model output for {label} data: {output[target_idx].cpu().detach().numpy()}\n")


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
    parser.add_argument('--data_csv', type=str, default='../../data/Combined_CSVs/combined_000-030.csv', help='Path to the data file.')
    return parser.parse_args()

if __name__ == '__main__':
    main()