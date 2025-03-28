# SPARCS Test

This repository contains some PyTorch code to test the capabilities of the novel SPARCS paradigm for NAS (*Neural Architecture Search*). The code has been written with simplicity in mind, prioritizing clarity over efficiency.

A **lighter** version of this code is available in notebook form at [this link](https://colab.research.google.com/drive/1PJeW-4dCKZ9oDNq8t3-yHUsZdGbOJF2y?usp=sharing).

The linked notebook is optimized to run on Colab's free tier by scaling down the experiment's parameters considerably. As a result, the outcomes achieved with the notebook will not match the quality of those obtained using a local installation of this repository's code.

## How to create the environment to run the code

The installation instructions assume a Linux system with an Nvidia GPU, though `config.yaml` provides an option to run the code on CPU.

In the following we also assume the use of `conda` for environment management. The following commands should be run one by one after installing `miniconda`:

```bash
conda create -n "sparcs" python=3.12.0
conda activate sparcs
pip install ray
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install matplotlib
pip install pyaml
pip install tqdm
pip install ipykernel
pip install ipywidgets
pip install scipy
```

Keep in mind that the link in `pip install torch --index-url https://download.pytorch.org/whl/cu124` could be dependent on your card, check the [PyTorch site](https://pytorch.org/) for more informations.

## How to run the code

### Create the datasets

First of all you should create the training datasets by simply running `generate_datasets.py`.

### Simple run

You can train the models by running the `train_models.py` script. Since the training can take some time `run_in_background.sh` is also available to run the same script in background:

```bash
./run_in_background.sh train_models.py
```

if the script doesn't work try running `chmod +x` on it first.

### Running in parallel

**Attention:** before running check the compatibility between your system's specs and the parameters present in the `run_train_in_parallel.py` script.

To accelerate the training you can run the code in parallel using `ray` with:

```bash
python run_train_in_parallel.py
```

This script can also be executed in the background:

```bash
./run_in_background.sh run_train_in_background.py
```

### Modifying the experiment parameters

To change the experiment parameters, simply modify the `config.yaml` file. The default parameters match those used to generate the results presented in the paper.

## The extra notebooks

This project also includes two extra notebooks: the first one, `show_scaling_law.ipynb`, simply carries out the steps to produce one of the graphs present in the paper. The second one, dubbed `extra_experiment.ipynb`, is more important: it carries out the *second test* reported in the paper, and you can simply run it like any other jupyter notebook.

## Acknowledgments

We would like to thank the following institutions for their support:

<p float="left">
    <a href="https://www.unifi.it/en"><img src="https://upload.wikimedia.org/wikipedia/commons/b/b4/Logo_unifi.jpg" height="100" /></a>
    <a href="https://home.infn.it/en/"><img src="https://www.infn.it/wp-content/uploads/2017/06/LOGO_INFN_NEWS_sito.jpg" height="100" /></a>
    <a href="https://next-generation-eu.europa.eu/index_en"><img src="https://commission.europa.eu/sites/default/files/styles/oe_theme_medium_no_crop/public/2022-11/next_gen_eu_logo_210611_360_2403.jpg?itok=kITbDc5L" height="100" /></a>
</p>
