# pyMEGA <a href='https://github.com/OSU-BMBL/pyMEGA'><img src='img/logo.png' align="right" height="139" /></a>

**Warning: This repository is under heavy development and the content is not final yet.**

pyMEGA is a deep learning package for identifying cancer-associated tissue-resident microbes.

If you have any questions or feedback, please contact Qin Ma <qin.ma@osumc.edu>.

The package is also available on PyPI: https://pypi.org/project/pyMEGA/

## News

### v0.0.1 - 1/24/2023
Added:
1. GitHub published: https://github.com/OSU-BMBL/pyMEGA
2. PyPI published: https://pypi.org/project/pyMEGA/

## Dev environment

pyMEGA is developed and tested in the following software and hardware environment:

```{bash}
python: 3.7.12
PyTorch: 1.4.0
NVIDIA Driver Version: 450.102.04
CUDA Version: 11.6
GPU: A100-PCIE-80GB
System: Red Hat Enterprise Linux release 8.3 (Ootpa)
```

## Installation

The following packages and versions are required to run pyMEGA:

- python: 3.7+
- cuda: 10.2
- torch==1.4.0 (must be 1.4.0)
- torch-cluster==1.5.4
- torch-geometric==1.4.3
- torch-scatter==2.0.4
- torch-sparse==0.6.1
- R > 4.0
- taxizedb (An R package for NCBI database)

Note: It is **highly suggested** to install the dependencies using [micromamba](https://mamba.readthedocs.io/en/latest/installation.html#install-script) (about 10 mins) rather than ```conda``` (could take more than 2 hours). If you don't want to use micromamba, just simply replace ```micromamba``` with ```conda``` in the code below.

**if you have GPU available: check [GPU version (CUDA 10.2)](#gpu-version-cuda-102)**

**if you only have CPU available: check [CPU version](#cpu-version)**

### GPU version (CUDA 10.2)

1. Create a virtual environment for pyMEGA

```{bash}
micromamba create -n pyMEGA_env python=3.7 -y
```

2. Activate ```pyMEGA_env```

```{bash}
micromamba activate pyMEGA_env
```

3. install ```pytorch v1.4.0```

```{bash}
micromamba install pytorch==1.4.0 cudatoolkit=10.1 -c pytorch -y
```

4. install other required packages from pip

```{bash}
pip install dill kneed imblearn matplotlib tqdm seaborn pipx
```

5. install ```torch-geometric for pytorch v1.4.0```

```{bash}
pip install torch-scatter==2.0.4 torch-sparse==0.6.1 torch-cluster==1.5.4 torch-spline-conv==1.2.0 torch-geometric==1.4.3 -f https://data.pyg.org/whl/torch-1.4.0%2Bcu101.html
```

6. install ```pyMEGA```
```{bash}
pip install pyMEGA
```

7. install ```R and taxizedb```

```{bash}
micromamba install R -y
```

8. verify the installation
```{bash}
pyMEGA -h
```

### CPU version

1. Create an virtual environment for pyMEGA

```{bash}
micromamba create -n pyMEGA_cpu_env python=3.7 -y
```

2. Activate ```pyMEGA_cpu_env```

```{bash}
micromamba activate pyMEGA_cpu_env
```

3. install ```pytorch v1.4.0```

```{bash}
#micromamba install pytorch==1.4.0 cpuonly -c pytorch -y
pip install torch==1.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

4. install other required packages from pip

```{bash}
pip install dill kneed imblearn matplotlib tqdm seaborn pipx
```

5. install ```torch-geometric for pytorch v1.4.0```

```{bash}
pip install torch-scatter==2.0.4 torch-sparse==0.6.1 torch-cluster==1.5.4 torch-spline-conv==1.2.0 torch-geometric==1.4.3 -f https://data.pyg.org/whl/torch-1.4.0%2Bcpu.html
```

6. install ```pyMEGA```
```{bash}
pip install pyMEGA
```

7. install ```R and taxizedb```

```{bash}
micromamba install R -y
```

8. verify the installation
```{bash}
pyMEGA -h
```


## Input data

### Data format

1. **Abundance matrix**: A CSV matrix. The first column represents the species IDs or official NCBI taxonomy names. The first row represents the sample names. pyMEGA will automatically try to convert the species name to IDs when needed.

![](./img/input_abundance.png)


2. **Sample labels**: A CSV matrix with a header row. The first column represents the species IDs or official NCBI taxonomy names. The first row represents the sample names. pyMEGA will automatically try to convert the species name to IDs when needed.

![](./img/input_metadata.png)

### Example data

```cre_abundance_data.csv```: The abundance matrix has 995 species and 230 samples

```cre_metadata.csv```: The sample labels of the corresponding abundance matrix. It has 230 rows (samples) and 2 columns

```NJS16_metabolic_relation.txt```: Human gut metabolic relationship database (reference: https://www.nature.com/articles/ncomms15393).

```{bash}
wget https://raw.githubusercontent.com/OSU-BMBL/pyMEGA/master/pyMEGA/data/cre_abundance_data.csv

wget https://raw.githubusercontent.com/OSU-BMBL/pyMEGA/master/pyMEGA/data/cre_metadata.csv

wget https://raw.githubusercontent.com/OSU-BMBL/pyMEGA/master/pyMEGA/data/NJS16_metabolic_relation.txt
```

## How to run pyMEGA

We will use the [example data](#example-data) for the following tutorial.

### Quick start

- ```input1```: the path to the abundance matrix
- ```input2```: the path to the sample metadata
- ```cuda```: which GPU device to use. Set to -1 if you only have CPU available

Running time:

- GPU version: about 15 mins
- CPU version: about 60 mins

#### GPU version

```{bash}
pyMEGA -cuda 0 -input1 cre_abundance_data.csv -input2 cre_metadata.csv -db NJS16_metabolic_relation.txt -o ./out
```

#### CPU version

```{bash}
pyMEGA -cuda -1 -input1 cre_abundance_data.csv -input2 cre_metadata.csv -db NJS16_metabolic_relation.txt -o ./out
```

### Enabling other parameters

use ```pyMEGA -h``` to check more details about parameters

```{bash}

INPUT1=cre_abundance_data.csv
INPUT2=cre_metadata.csv
DB=NJS16_metabolic_relation.txt
CUDA=0
LR=0.003
N_HID=128
EPOCH=30
KL_COEF=0.00005
THRES=3
OUTPUT=./out
pyMEGA -input1 ${INPUT1} -input2 ${INPUT2} -db ${DB} -epoch ${EPOCH} -cuda ${CUDA} -n_hid ${N_HID} -lr ${LR} -kl_coef ${KL_COEF} -o ${OUTPUT}

```

## Output files

1. ```*_final_taxa.txt``` : Cancer-associated microbal signatures. This is an unstructured txt file separated by tab. This is the final output file.

![](./img/output_tax.png)

2. ```*_taxa_num.csv``` : normalized attention score for each species under each cancel label 

3. ```*_metabolic_matrix.csv```: metabolic relationship network extracted from database

4. ```*_phy_matrix.csv```: phylogenetic relationship network extracted from NCBI taxonomy database

5. ```*_attention.csv```: raw attention matrix extracted from deep learning model

## Visualize results using Rcy3 and Cytoscape

Check the README file in ```./cytoscape``` folder:

[./cytoscape/README.md](./cytoscape/README.md)

## Acknowledgements

Maintainer: [Cankun Wang](https://github.com/Wang-Cankun)

Contributors:

- Cankun Wang
- Megan McNutt
- Anjun Ma
- Zhaoqian Liu
- Yuhan Sun

Contact us: Qin Ma <qin.ma@osumc.edu>.
