# SPARCS Test

This repo contains some PyTorch code to test the capability of the novel SPARCS paradigm for NAS (*Neural Architecture Search*).

A Colab notebook is also available at [this link](https://colab.research.google.com/drive/1PJeW-4dCKZ9oDNq8t3-yHUsZdGbOJF2y?usp=sharing).
Keep in mind that the code may run slow with the limited resources of the free Colab plan.

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
```

Keep in mind that the link in `pip install torch --index-url https://download.pytorch.org/whl/cu124` could be dependent on your card, check the [PyTorch site](https://pytorch.org/) for more informations.