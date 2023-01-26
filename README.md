# pyMEGA

**Warning: This repository is under heavy development and the content is not final yet.**

pyMEGA is a deep learning package for identifying cancer-associated tissue-resident microbes.

If you have any questions or feedback, please contact Qin Ma <qin.ma@osumc.edu>.

The pacakge is also available on PyPI: https://pypi.org/project/pyMEGA/

## News

## v0.0.1 1/24/2023
1. GitHub published: https://github.com/OSU-BMBL/pyMEGA
2. PyPI published: https://pypi.org/project/pyMEGA/

## Dev environment

pyMEGA is developed and tested in the following software and hardware environment:

```{bash}
python: 3.7.12
pytorch: 1.4.0
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

Note: It is **highly suggested** to install the dependencies using [micromamba](https://mamba.readthedocs.io/en/latest/installation.html#install-script) (about 10 mins) rather than ```conda``` (could take more than 2 hours)

**if you have GPU available: check [GPU version (CUDA 10.2)]()**

**if you only have CPU available: check [CPU version]()**

### GPU version (CUDA 10.2)

1. Create an virtual environment for pyMEGA

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

7. verify installation
```{bash}
pyMEGA -h
```

### CPU version

1. Create an virtual environment for pyMEGA

```{bash}
micromamba create -n pyMEGA_cpu_env python=3.7 -y
```

2. Activate ```pyMEGA_env```

```{bash}
micromamba activate pyMEGA_cpu_env
```

3. install ```pytorch v1.4.0```

```{bash}
micromamba install pytorch==1.4.0 -c pytorch -y
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

7. verify installation
```{bash}
pyMEGA -h
```


### Input data

#### Data format

The first column represents the species IDs or official NCBI taxonomy names. The first row represents the sample names. pyMEGA will automatically try to convert the species name to IDs when needed.


![](./img/input_abundance.png)

#### Example data

```cre_abundance_data.csv```: 995 species x 230 samples

```
wget cre_abundance_data.csv
```

## Acknowledgements

Maintainer: [Cankun Wang](https://github.com/Wang-Cankun)

Contributors:

- [Cankun Wang](https://github.com/Wang-Cankun)

Contact us: Qin Ma <qin.ma@osumc.edu>.