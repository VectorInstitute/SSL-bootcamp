# Self-Supervised Learning (SSL) Bootcamp
This repository contains reference implementations of three self-supervised learning
techniques explored during the Vector Institute's Self-Supervised Learning (SSL) Bootcamp.

# Summary of Reference Implementations

| Name | Description | Reference Implementation  |
|------|-------------|-------|
Internal Contrastive Learning (ICL) + Latent Outlier Exposure (LOE)| ICL learns to maximize the mutual information between two complementary subsets based on the assumption that the relation between a subset of features and the rest of the features is class-dependent. LOE extends ICL to work with contaminated datasets. | [Anomaly Detection in Tabular Data with ICL](src/contrastive_learning/ICL/ICL.ipynb), [Latent Outlier Exposure for Anomaly Detection with Contaminated Data](src/contrastive_learning/LatentOE/LatentOE_Notebook.ipynb)
SimMTM | Reconstructs a time series signal from multiple randomly masked versions. Uses series-wise representation similarity to do a weighted aggregation of point-wise representations before reconstruction. | [Beijing PM2.5 Air Quality Forecasting](src/masked_modelling/simmtm/simmtm-BeijingPM25Quality-forecasting.ipynb)
TabRet | TabRet is a pre-trainable Transformer-based model for tabular data and designed to work on a downstream task that contains columns not seen in pre-training. Unlike other methods, TabRet has an extra learning step before fine-tuning called retokenizing, which calibrates feature embeddings based on the masked autoencoding loss. | [Stroke Prediction with the BRFSS dataset](src/masked_modelling/tabret/TabRet.ipynb)
Data2Vec | Combines masked prediction with self-distillation to predict contextualized latent representations (produced by the teacher network) based on a partial/masked view of the input (given to the student network). | [Image Classification with STL-10 dataset](src/self_distillation/data2vec_vision.ipynb)


# Setting up the environment
Prior to installing the dependencies for this project, it is recommended to install
[uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) and create
a virtual environment. You may use whatever virtual environment management tool
that you like, including uv, conda, and virtualenv.

With uv, you can create a virtual environment with the following command:

```bash
uv venv -n --seed --python 3.9 /path/to/new/virtual/environment/ssl_env`
```
This will create a new virtual environment in the specified path.

**Note**: If you are using the Vector Institute's Vaughan cluster, a virtual
environment has already been created for you at `/ssd003/projects/aieng/public/ssl_bootcamp_resources/venv`.

Once you have created a virtual environment, you can activate it with the command:

```
source /path/to/new/virtual/environment/ssl_env/bin/activate
```

Then, you can install the dependencies for this project with the following command:

```bash
git clone https://github.com/VectorInstitute/SSL-Bootcamp.git
cd SSL-Bootcamp
uv sync --no-cache --active --dev
```
**Note**: The `--active` flag in the above command assumes that you have already
activated your virtual environment. If you prefer not to create a new virtual
environment yourself, you can omit the `--active` flag and uv will create a new virtual environment
for you in the `.venv` directory inside the project root.

## Using pre-commit hooks
To ensure that your code adheres to the project's style and formatting guidelines,
you can use pre-commit hooks to check for common issues, such as code formatting,
linting, and security vulnerabilities. Run the following command before pushing
your code to the repository:

```
pre-commit run --all-files
```
