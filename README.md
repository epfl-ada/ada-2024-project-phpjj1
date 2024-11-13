
# Project-phpjj1
This is the ADA 2024/25 project repository for the group phpjj1.

## Quickstart

```bash
# clone project
git clone <project link>
cd <project repo>

# [OPTIONAL] create conda environment
conda create -n <env_name> python=3.11 or ...
conda activate <env_name>
# Necessary if using a conda environment otherwise pip will install packages globally even if the conda environment is active
conda install pip


# install requirements
pip install -r pip_requirements.txt
```



### How to use the library
Tell us how the code is arranged, any explanations goes here.



## Project Structure

The directory structure of our project looks like this:

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- Our result notebook
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```

## Downloading Raw Datasets

**Primary Dataset**: [CMU Movie Summary Corpus](https://www.cs.cmu.edu/~ark/personas/) (`MovieSummaries`)

#### Secondary Datasets:
- [MovieLens 32M Dataset](https://grouplens.org/datasets/movielens/#:~:text=for%20new%20research-,MovieLens%2032M,-MovieLens%2032M%20movie) (`ml-32m`)


For each dataset, download the zip file and place it in the root level `data` directory. Do NOT rename the unzipped folder.

Create folder `data/processed` to store the processed datasets. You need this or else the merging script will fail.
