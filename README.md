# Construction de tables de mortalité prospectives d'expérience pour des garanties de prévoyance au Royaume-Uni et en Irlande

## Structure

```
.
├── data # directory to store input portfolios (in .csv format)
├── HMD_inputs # directory to store HMD tables (in .txt format)
│   └── matrices # directory where processed HMD files into .csv are stored
├── images # directory where figure are stored as images
├── matrices # directory where mortality tables are stored as .csv files
├── tables # directory to store output latex tables
├── bongaarts.ipynb # bongaarts (Planchet + Optuna methods)
├── bongaarts_studies.ipynb # bongaarts model (only Optuna method)
├── planchet.ipynb # bongaarts model (only Planchet method)
├── brass.ipynb # brass model
├── fumeurs.ipynb # smokers
├── hmd.ipynb # HMD displays
├── smoothing.ipynb # Whittaker-Henderson smoothing
├── README.md
└── utils # utility functions
    ├── __init__.py
    ├── studies.py # Optuna utilities
    └── utils.py
```
