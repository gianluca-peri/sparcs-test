# SPARCS Test

This repo contains some PyTorch code to test the capability of the novel SPARCS paradigm for NAS (*Neural Architecture Search*). The coda has been written with the goal of making the process as simple to understand as possible, and for now no attention has been devolved to making it efficient.

A lighter version of this code is also available in notebook form at [this link](https://colab.research.google.com/drive/1PJeW-4dCKZ9oDNq8t3-yHUsZdGbOJF2y?usp=sharing).

The linked notebook is light enough to be runnable with the limited resources of Colab's free tier, but to achive this we scaled down the experiment's parameters considerably, and so the results achievable with the notebook will not have the same quality as the one obtainable with a local installation of this repo's code.

## How to create the environment to run the code

It is assumed installation on linux, as well as availability of an Nvidia card.
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