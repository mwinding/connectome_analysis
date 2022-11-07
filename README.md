connectome_analysis
==============================Scripts to manipulate, process, and analyze the Drosophila larval brain connectome

Installation and setup
--------
The associated tools in connectome_tools_deprecated have been reimplemented in the connectome_tools repo with updated features, https://github.com/mwinding/connectome_tools

connectome tools can be installed via pip
``pip install git+https://github.com/mwinding/connectome_tools``

Project Organization
------------
├── LICENSE
├── README.md
├── data
│   ├── adj             <- raw connectivity matrices, including input/output counts
│   ├── cascades        <- signal cascade datasets
│   ├── edges_threshold <- edge lists based on pair-wise thresholding of /adj
│   ├── graphs          <- saved graphs and meta_data.csv
│   ├── pairs           <- pair list of homologous left/right neuron pairs
│   └── processed       <- processed raw data from CATMAID, used to generate /adj
│
├── plots               <- where all PDF plots are saved
│
├── generate_data       <- scripts to generate datasets in /data folder
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── scripts            <- scripts folder, containing 
│   │
│   ├── network_analysis
│   │   └── cascade_hubs_by_degree.py
|   |   └── etc.
│   │
│   └── etc.
│       └── etc.
|
└── scripts-for-figures.py  <- not currently updated

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>