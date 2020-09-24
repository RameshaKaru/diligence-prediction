## Setting the environment

- Install anaconda following [these instructions](https://docs.anaconda.com/anaconda/install/linux/) 
- Navigate to the root project folder. (Below commands are given assuming executions are done from the root package)
- Execute the below command to install needed dependencies
```commandline
 conda env create -f setup/environment.yaml
```
- Then you can activate the conda environment using below command

```commandline
conda activate hdpsenv
```
- Needed R packages can be installed by executing the below command.

```commandline
python setup/setup_r_lib.py
```

- Choose the needed CRAN mirror when prompted. Recommended to use the 1st mirror.

Now the environment is ready with all needed dependencies.

> Main documentation [link](README.md)

