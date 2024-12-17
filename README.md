# ITU BDS SDSE'24 - Project by group GOMAD🥛

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://i.ytimg.com/vi/tEOTC7IDAew/maxresdefault.jpg" />
</a>

This is the repository of 2025 Software Development course project.

## Project Organization

```
├── LICENSE            <- Open-source license
├── Makefile           <- Makefile with extra documentation
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         sdw_project and configuration for tools like black
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── scripts   <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes sdw_project a Python module
    │
    ├── deployment.py  <- Handles deployment configurations and variables
    │    
    ├── evaluation.py  <- Contains code for evaluating models
    │    
    ├── features.py    <- Scripts for feature engineering
    │    
    ├── get_data.py    <- Scripts to download or retrieve data
    │    
    ├── mlflow_client.py <- Is used so that other scripts can use the same MLFlow client
    │
    ├── preprocessing.py <- Scripts to preprocess data
    │
    ├── train.py       <- Code to train machine learning models
    │

 
```

--------

To run the code, paste this into the command line:
```console
go run main.go
```
