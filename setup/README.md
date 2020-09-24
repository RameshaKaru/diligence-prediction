### Setting the environment

- Install anaconda following [these instructions](https://docs.anaconda.com/anaconda/install/linux/) 
- Execute the below command to install needed dependencies
```commandline
 conda env create -f environment.yaml
```
- Then you can activate the conda environment using below command

```commandline
conda activate hdpsenv
```
- Needed R packages can be installed by executing the below command. (Assuming execution is done from the root package)

```commandline
python setup/setup_r_lib.py
```

- Choose the needed CRAN mirror when prompted. Recommended to use the 1st mirror.

Now the environment is ready with all needed dependencies.