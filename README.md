# Self-Supervised Learning (SSL) Bootcamp
This repository contains reference implementations of three self-supervised learning
techniques explored during the Vector Institute's Self-Supervised Learning (SSL) Bootcamp.

# Installing dependencies
```
python3 -m venv /path/to/new/virtual/environment/ssl_env
source /path/to/new/virtual/environment/ssl_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you are on the Vector Institute's Vaughan cluster, the environment is already set up and can be activated with

```
source /ssd003/projects/aieng/public/ssl_bootcamp_resources/venv/bin/activate
```

# Using pre-commit hooks
To check your code at commit time
```
pre-commit install
```

You can also get pre-commit to fix your code
```
pre-commit run --all-files
```
