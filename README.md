# human-pose3d

## data_explorer.py

- skeleton format:
+ MPII : http://human-pose.mpi-inf.mpg.de/#download
+ LSP : http://sam.johnson.io/research/lsp.html

directory tree:
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
    
- label.pkl: store list of dict, for every dict:
+ 'index': eg. '00000' 
+ 'joints': (3, 14)
+ 'ordinal': (14, 14)

**Note** MPII dataset has been reformated by `data_explorer.py`, so has the same nodes as LSP(14)
