# SPARCS Test

This repository contains some PyTorch code to test the capabilities of the novel SPARCS paradigm for NAS (*Neural Architecture Search*). The code has been written with simplicity in mind, prioritizing clarity over efficiency.

A **lighter** version of this code is available in notebook form at [this link](https://colab.research.google.com/drive/1PJeW-4dCKZ9oDNq8t3-yHUsZdGbOJF2y?usp=sharing).

The linked notebook is optimized to run on Colab's free tier by scaling down the experiment's parameters considerably. As a result, the outcomes achieved with the notebook will not match the quality of those obtained using a local installation of this repository's code.

## How to create the environment to run the code

The installation instructions assume a Linux system with an Nvidia GPU.
First of all you should run `nvidia-smi` to check the specs of your card.

In the following we also assume `conda` usage; it is advisable to run these shell instructions one by one, after having installed `miniconda`

```bash
conda create -n "sparcs" python
conda activate sparcs
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install matplotlib
pip install pyaml
```

Keep in mind that the link in `pip install torch --index-url https://download.pytorch.org/whl/cu124` could be dependent on your card, check the [PyTorch site](https://pytorch.org/) for more informations.

## How to run the code

You can train the models by running the `train_models.py` script. Since the training can take some time `run_train_in_background.sh` is also available to run the same script in background, logging the output (just remember to `chmod +x` if it doesn't work).

If you wish to change the parameters of the experiment you can do so by simply modifing the `config.yaml` file; currently the parameters are the same as the ones used to get the results presented in the paper.

## Acknowledgments

We would like to thank the following institutions for their support:

<p float="left">
    <a href="https://www.unifi.it/en"><img src="https://upload.wikimedia.org/wikipedia/commons/b/b4/Logo_unifi.jpg" height="100" /></a>
    <a href="https://home.infn.it/en/"><img src="https://home.infn.it/images/news/LOGO_INFN_NEWS_sito.jpg" height="100" /></a>
    <a href="https://next-generation-eu.europa.eu/index_en"><img src="https://commission.europa.eu/sites/default/files/styles/oe_theme_medium_no_crop/public/2022-11/next_gen_eu_logo_210611_360_2403.jpg?itok=kITbDc5L" height="100" /></a>
</p>
