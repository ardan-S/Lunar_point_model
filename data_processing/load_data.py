import numpy as np
import pandas as pd

import os
import sys

sys.path.append(os.path.abspath('.'))

from process_image import download_parse_metadata, process_image
from process_urls import get_file_urls, process_url, process_urls_in_parallel


Diviner_home = 'https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/data_derived_gdr_l3/2016/polar/jp2/'
LOLA_home = 'https://imbrium.mit.edu/DATA/LOLA_GDR/POLAR/JP2/'

Diviner_urls = get_file_urls(Diviner_home, '.lbl', 'tbol')
LOLA_urls = get_file_urls(LOLA_home, '.LBL', 'LDRM')

Diviner_urls = Diviner_urls[:2]
LOLA_urls = LOLA_urls[:2]

output_csv_path = 'All_data.csv'

print("Begin processing data...")
Diviner_df = process_urls_in_parallel(Diviner_urls, output_csv_path, 'Diviner')
print("Diviner data processed.")
LOLA_df = process_urls_in_parallel(LOLA_urls, output_csv_path, 'LOLA')
print("LOLA data processed.")
print("All data processed.")
# Merge DataFrames on 'Longitude' and 'Latitude'
combined_df = pd.merge(Diviner_df, LOLA_df, on=['Longitude', 'Latitude'], how='outer')
print("Data combined into single df.")
print(combined_df.head(10))