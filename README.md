# Artificial-Judge
Artifical Judge attempts to use the BERT NLP pre-trained model to predict the winners of supreme court cases. This notebook uses [this kaggle dataset](https://www.kaggle.com/datasets/deepcontractor/supreme-court-judgment-prediction)
which can be found in the justice.csv file in this repo. 

## Setup
To begin, install conda or miniconda for your machine: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html      
Install jupyter lab as well: 
```
conda install jupyterlab -c conda-forge
```
Next, create a conda env to run this notebook in: 
```
conda create --name myenv python=3.7
```
run `conda activate myenv` to enter in to the env. Note: run `conda deactivate` to go back to base

Prepare a kernel for jupyter lab in your environment:       
- Activate the environment: `source activate myEnv`       
- Install ipykernel: `conda install ipykernel`     
- Run this command: `python -m ipykernel install --user --name myEnv --display-name “my_environment”`

Check import in the notebook for all necessary packages. You can check to see what packages are installed in your current conda env with `conda list`. If you are missing a package, simply run `conda install packagename`

