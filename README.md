## OSSC

This repository contains the implementation code of paper "A Dynamic Variational Framework for Open-world Node Classification in Structured Sequences".


## Dependencies

- Python (>=3.6)
- torch:  (>= 1.7.0)
- numpy (>=1.17.4)
- sklearn



## Implementation

Here we provide the implementation of OSSC. The repository is organised as follows:

 - `data/` contains the necessary dataset files and config files;
 - `Method/` contains the implementation of the OSSC and the basic utils;

 Finally, `main.py` puts all of the above together and can be used to execute a full training run on the datasets.

## Process
 - Place the datasets in `data/`
 - Training/Testing:
 ```bash
 python main.py
 ```
 
