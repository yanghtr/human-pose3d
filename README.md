# human-pose3d
dataset.py

## dataset.py
```
python dataset.py
```
.
├── dataset
│   ├── images
│   ├── label_mpii_lsp.pkl
│   └── LICENSE
└── dataset.py


## data_explorer.py (ignore, just used to explore data)

- skeleton format:
    - MPII : http://human-pose.mpi-inf.mpg.de/#download
    - LSP : http://sam.johnson.io/research/lsp.html

directory tree:
```Python
    directory tree
    ./lsp-mpii-ordinal
    ├── data_explorer.py
    ├── lsp_dataset_original
    │   ├── images
    │   ├── joints.mat
    │   ├── ** label.pkl
    │   ├── LICENSE
    │   ├── ordinal.mat
    │   └── README.txt
    ├── mpii_upis1h
    │   ├── images
    │   ├── joints.mat
    │   ├── ** label.pkl
    │   ├── LICENSE
    │   └── ordinal.mat
    └── README
```
- label.pkl: store list of dict, for every dict:
```
'index': eg. '00000' 
'joints': (3, 14)
'ordinal': (14, 14)
```
**Note** MPII dataset has been reformated by `data_explorer.py`, so has the same nodes as LSP(14)


