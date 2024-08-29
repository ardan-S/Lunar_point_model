# Using deep learning to predict water ice presence at the lunar poles

<p>
<strong>- Independent Research Project by Ardan Suphi (as5023)</strong><br>
<strong>- MSc in Applied Computational Science and Engineering</strong><br>
<strong>- Department of Earth Science and Engineering, Imperial College London</strong>
</p>


## Table of Contents

1. [Project overview](#overview)
2. [File structure](#structure)
3. [Setup](#setup)
4. [Usage](#usage)

<a name="overview"></a>
## Project overview
The presence, location, and abundance of volatile substances, such as water ice, on the lunar surface have been subjects of long-standing scientific inquiry and many consider the most likely locations for them as inside permanently shadowed regions (PSRs). Remote sensing has emerged as a valuable tool for estimating these distributions.

This project proposes that the consolidation of several of these remote sensing datasets and the application of machine learning (ML) algorithms can improve the estimates of the presence and
distribution of volatile substances.

The overall goal of this project was to collect, combine and label several of NASA’s remote sensing datasets, then train a machine learning model to score coordinate points on the lunar surface for the presence of voliatiles, based on remote sensing indicators.

As inputs, the model will receive coordinates plus remote sensing values for Diviner, LOLA, M<sup>3</sup> and Mini-RF, and output a score between 0 and 7 with the former representing absolutely no indication of volatile material and the later representing the maximum possible indication of such materials.

Datasets were decoded, interpolated and synthetically labelled. An example of this process plus the distribution of the labels can be seen in images (a) to (c).

<p align="center">
    <img src="data/M3/M3.png" alt="Raw Image" width="500"/>
    <br><strong>(a)</strong> Plot of raw M<sup>3</sup> data
</p>

<p align="center">
    <img src="data/M3/M3_interp.png" alt="Interpolated Image" width="500"/>
    <br><strong>(b)</strong> Plot of interpolated M<sup>3</sup> data
</p>

<p align="center">
    <img src="data/Combined_data.png" alt="Label Image" width="500"/>
    <br><strong>(c)</strong> Plot of the labels. A label of 0 represents all datasets showing no indication, a label of 7 represents all datasets showing the maximum possible indication.
</p>


Following this data processing, two neural networks were made:
- A fully connected neural network with self attention and a residual connection (FCNN)
- A graph convolutional network with a residual connection (GCN)

These were then trained, tuned and saved to produce the final model which could predict labels at any coordinate with the remote sensing data. 

<a name="structure"></a>
## File structure

    ├── data/
    │   ├── Combined_CSVs/
    │   ├── Diviner-temp/
    │   ├── LOLA-Albedo/
    │   ├── M3/
    │   └── Mini-RF/
    ├── data_processing
    │   ├── logs/
    │   ├── interp.pbs
    │   ├── interpolate.py
    │   ├── label.pbs
    │   ├── label.py
    │   ├── plot.pbs
    │   ├── plot.py
    │   ├── process_image_dask.py
    │   ├── process_urls.dask.py
    │   └── utils_dask.py
    ├── deliverables
    ├── lit_review
    ├── model/
    │   ├── figs/
    │   ├── logs/
    │   ├── model_scripts/
    │   └── src/
    │       ├── evaluate.py
    │       ├── FCNN_hypertune.py
    │       ├── GCN_hypertune.py
    │       ├── models.py
    │       ├── run_model.py
    │       ├── trainFCNN.py
    │       ├── trainGNN.py
    │       └── utils.py
    ├── title/
    ├── environment.yml
    ├── LICENSE
    ├── README.md
    └── requirements.txt

<a name="setup"></a>
## Setup

### Prerequisites
- **Git**: Ensure you have Git installed on your machine.
- **Python**: Python 3.11.5 reccomended, Python 3.8 minimum
- **Conda** (optional but recommended): If using `environment.yml`, having Anaconda or Miniconda installed is recommended.

### Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/ese-msc-2023/irp-as5023.git
cd irp-as5023
```

### Setting Up the Environment

You can set up the environment using one of the following methods:

#### 1. Using `requirements.txt` with `pip`
If you prefer to use `pip`, follow these steps:

1. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  
    # On Windows use `venv\Scripts\activate`
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

#### 2. Using `environment.yml` with Conda
If you are using Conda, follow these steps:

1. **Create the environment from `environment.yml`**:
    ```bash
    conda env create -f environment.yml
    ```

2. **Activate the environment**:
    ```bash
    conda activate your_environment_name
    ```

3. **(Optional) Update the environment** if there are any changes:
    ```bash
    conda env update -f environment.yml
    ```

### Final Steps

After setting up the environment, your project should be ready to use. Follow any additional instructions in the "Usage" section to start working with the project.

<a name="usage"></a>
## Usage
### To obtain the training data:
CSVs are not provided in this repository and must be generated and saved locally. The `evaluate.pbs` script can be used to fully process images and save them as CSVs. 


The `download.py` file is provided to download image and metadata files in advance of processing them. This is advised for M<sup>3</sup> and Mini-RF datasets as their large size causes internet connectivity to significantly affect processing time. This was not observed to impact processing of Diviner and LOLA which can be processed directly from the web using `evaluate.pbs`. 

Having generated CSVs for each dataset, run the interpolation and then label jobscripts to generate the respective CSVs, including the final, concatenated and labelled dataset. 

### To train the models:
Jobscripts to train the FCNN and GCN are provided in `model/` directory. Hyperparameters can be adjusted in the jobscript and the model and parameters will be saved unless this argument is adjusted in the source file. Note the training data must be obtained in the steps above to train the models. 

### To use the saved models:
The `run_model.pbs` jobscript can be used to load saved models and evaluate points. It is currently set to choose a random point from the interpolated data but can easily be modified to return a value on an external input. Note that the FCNN can input a single point with the associated remote sensing values for a label but the GCN requires a graph object as an input and will output labels for each node. 