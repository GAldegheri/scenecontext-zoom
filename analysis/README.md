# Data analysis code

This code was tested using Python 3.7.10. The dependencies can be found in [requirements.txt](./requirements.txt).

To setup a Conda environment to run this code:

```
conda create -n "scenecontext" python=3.7.10
conda activate scenecontext
pip install -r requirements.txt
```

The notebook [ReadData.ipynb](./ReadData.ipynb) contains all the code necessary to reproduce results and plots in the paper.

Utilities to read and analyze the data can be found in [read_data.py](./read_data.py). While we only include preprocessed data, code to extract the raw data from online experiments (for anyone interested in running these experiments themselves) can be found in [extract_raw_data.py](./extract_raw_data.py).
