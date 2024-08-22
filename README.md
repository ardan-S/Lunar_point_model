# Using deep learning to map potential resource distribution of lunar permanently shadowed regions

<p>
<strong>- Independent Research Project by Ardan Suphi (as5023)</strong><br>
<strong>- MSc in Applied Computational Science and Engineering</strong><br>
<strong>- Department of Earth Science and Engineering, Imperial College London</strong>
</p>


## Table of Contents

1. [Project overview](#overview)
2. [Project components](#components)
3. [File structure](#structure)
4. [Setup](#setup)
5. [Usage](#usage)

<a name="overview"></a>
## Project overview
The presence, location, and abundance of volatile substances, such as water ice, on the lunar surface have been subjects of long-standing scientific inquiry and many consider the most likely locations for them as inside permanently shadowed regions (PSRs). Remote sensing has emerged as a valuable tool for estimating these distributions.

This project proposes that the consolidation of several of these remote sensing datasets and the application of machine learning (ML) algorithms can improve the estimates of the presence and
distribution of volatile substances.

The overall goal of this project was to collect, combine and label several of NASA’s remote sensing datasets, then train a machine learning model to score coordinate points on the lunar surface for the presence of voliatiles, based on remote sensing indicators.

As inputs, the model will receive coordinates plus remote sensing values for Diviner, LOLA, M<sup>3</sup> and Mini-RF, and output a score between 0 and 7 with the former representing absolutely no indication of volatile material and the later representing the maximum possible indication of such materials.

Datasets were decoded and interpolated and synthetically labelled. The distribution of the labels can be seen in the image below.

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

<a name="components"></a>
## Project components

<a name="structure"></a>
## File structure

<a name="setup"></a>
## Setup
clone the repo
create env
data access

<a name="usage"></a>
## How to use the model / usage examples 