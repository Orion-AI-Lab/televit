# TeleViT: Teleconnection-driven Transformers Improve Subseasonal to Seasonal Wildfire Forecasting

This is the official code repository of the [TeleViT paper](https://arxiv.org/abs/2306.10940), accepted at ICCV 2023 [hadr.ai workshop](https://www.hadr.ai/).

🌐 Paper Website: https://orion-ai-lab.github.io/televit/

![televit_architecture](/docs/static/images/architecture.png)
Figure: TeleViT Architecture combining local input, with global input and teleconnection indices.

Authors: [Ioannis Prapas](https://iprapas.github.io) (1, 3), [Nikolaos-Ioannis Bountos](https://ngbountos.github.io/) (1, 2), 
[Spyros Kondylatos](https://github.com/skondylatos/) (1, 3), [Dimitrios Michail](https://github.com/d-michail) (3), [Gustau Camps-Valls](https://www.uv.es/gcamps/) (2), [Ioannis Papoutsis](https://scholar.google.gr/citations?user=46cBUO8AAAAJ) (1)

(1) Orion Lab, IAASARS, National Observatory of Athens

(2) Department of Informatics and Telematics, Harokopio University of Athens

(3) Image & Signal Processing (ISP) group, Universitat de València

## Prerequisites

Before running the code, you need to install the requiremetns, download the data, preprocess them and create the .env file.

The code uses ashleve's pytorch lightning hydra template https://github.com/ashleve/lightning-hydra-template. It is worth reading the [template's README](./README_template.md) before trying to run the code.

### Install requirements

Install the requirements in [requirements.txt](./requirements.txt).

### Download the data

Download the [SeasFire dataset](https://zenodo.org/record/8055879) from zenodo. Note it is 44GB. 

Unzip the dataset to a folder of your choice. We will refer to the unzipped zarr as `DATASET_PATH` in the rest of the README.

### Create the coarsened dataset

See this [notebook](notebooks/create_coarsened_cube.ipynb) on how to create the coarsened dataset. This is necessary for the TeleViT experiments.

We will refer to the coarsened dataset as `DATASET_PATH_GLOBAL` in the rest of the README.

### Create a `.env` file

Needs wandb account. If you don't want to use wandb, you need to dig into the code remove callbacks to wandblogger and use a different logger.

Create a `.env` file with the following variables:

```
WANDB_NAME_PREFIX="prefix_for_wandb_run_name"
WANDB_ENTITY="entity_for_wandb"
WANDB_PROJECT="project_for_wandb"
DATASET_PATH="`DATASET_PATH`"
DATASET_PATH_GLOBAL="`DATASET_PATH_GLOBAL`"
```

## Running the experiments

To run the U-Net baseline:

```
bash scripts/unet_experiments.sh
```

To run the TeleViT experiments:

```
bash scripts/televit_experiments.sh
```

## Note on Resources needed

- RAM memory: The code uses about 100GB of RAM to load the dataset into memory. This makes the process slow to start (waiting to preprocess the dataset and load it). However, it allows for the flexibility to change the ML dataset between runs, apply any kind of preprocessing, forecasting in different time horizons, adding/removing variables. 

- GPU memory: The code has been tested with NVIDIA GPUs with at least 24GB RAM. For smaller GPUs, you might need to play with the datamodule.batch_size

## Citation

If you use this code, please cite the following paper:

```
@article{prapas2023televit,
  title={TeleViT: Teleconnection-driven Transformers Improve Subseasonal to Seasonal Wildfire Forecasting},
  author={Prapas, Ioannis and Bountos, Nikolaos Ioannis and Kondylatos, Spyros and Michail, Dimitrios and Camps-Valls, Gustau and Papoutsis, Ioannis},
  journal={arXiv preprint arXiv:2306.10940},
  year={2023}
}
```
## Acknowledgements

This repo uses ashleve's pytorch lightning hydra template https://github.com/ashleve/lightning-hydra-template. 

This work is part of the SeasFire project, which deals with
”Earth System Deep Learning for Seasonal Fire Forecasting”
and is funded by the European Space Agency (ESA) in the
context of the ESA Future EO-1 Science for Society Call.