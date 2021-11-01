# Test Updated Containers

Last updated: Oct 21st 2021

This is for testing updated containers on Gradient, e.g., the ones shown to the user when creating a Notebook such as TensorFlow or PyTorch.

The emphasis here is not full-stack QA on the containers, but sanity-checking their use for data science *on Gradient*, in particular, with our GPUs.

So we have, as things to test

 - Container and software of interest sees the GPU
 - ML models pick up the GPU
 - Other software is the expected versions

and we want to test them on all 3 main interfaces

 - Notebooks: Jupyter Notebook `.ipynb`
 - Workflows: Python script `.py`
 - Deployments / general: command line

This gives other requirements too, e.g., install then `import gradient` should work without breaking things, as this is needed for the SDK, and we shouldn't have containers where the SDK is broken.

## Current files

In general each container will have its own software stack, so the contents of the testing scripts will vary.

So far we are testing updating the main TensorFlow and PyTorch containers from 2.4.1 and 1.8 to 2.6.0 and 1.10, giving

|Current container|New one to be tested|Interface|File|Works?|
|---|---|---|---|---|
|`tensorflow/tensorflow:2.4.1-gpu-jupyter`|`tensorflow/tensorflow:2.6.0-gpu-jupyter`|Notebook|`tensorflow-2.6.0-gpu-jupyter.ipynb`|Yes|
| | |Workflow|`tensorflow-2.6.0-gpu-jupyter.py`|Yes|
| | |CLI|`tensorflow-2.6.0-gpu-jupyter.sh`|Yes|
|`nvcr.io/nvidia/pytorch:21.02-py3`|`nvcr.io/nvidia/pytorch:21.09-py3`|Notebook|`pytorch-21.09-py3.ipynb`|**No (see Issues Found, below)**|
| | |Workflow|`pytorch-21.09-py3.py`|**No (see Issues Found, below)**|
| | |CLI|`pytorch-21.09-py3.sh`|Yes|

In general the `.ipynb` and `.py` will run the same commands, but the `.ipynb` has them separated into cells. Sometimes code will work in `.ipynb` but not `.py` or vice versa, e.g., the `.py` needs to use the subprocess module to call the command line.

## To run

A new Notebook needs to be started because some tests involve adding new software, e.g., `!pip3 install gradient`.

 - Create a Project
 - Start a new Gradient Notebook
 - Select instance type
   - In Notebook Advanced options:
     - Under container paste the name of the new container to be tested
     - Leave other options blank, e.g., Workspace -> Workspace URL, and Container -> Command
 - Start Notebook
 - Run `.sh` from Notebook terminal: `sh <filename>.sh` [1,2,3]
 - Run `.py` from same: `python <filename>.py`
 - Run `.ipynb` as notebook [4]
 
[1] Use the Jupyter/JupyterLab terminal, not the Gradient one, until it works better  
[2] In the TF container, the terminal works better if you type `bash` first after opening it  
[3] In JupyterLab on the PyTorch container, needs space/q to be pressed to proceed with output  
[4] The `.ipynb` was run in Jupyter/JupyterLab interface; not tried Gradient interface
 
## ISSUES FOUND: To be resolved by Product/QA

1. Installing `gradient` in the PyTorch Notebook (`!pip3 install gradient`) breaks PyTorch because gradient downgrades NumPy from 1.21 to 1.18. PyTorch then fails with

`RuntimeError: Numpy is not available`

Attempting to re-upgrade NumPy back to 1.21 causes `gradient` to give an error

`gradient-utils 0.3.2 requires numpy==1.18.5, but you have numpy 1.21.3 which is incompatible.`

-> `gradient`'s usage of versions needs to be updated to be compatible with PyTorch for this container to work.

2. Also, PyTorch data loading fails with

`ImportError: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html`

but `!pip3 install ipywidgets` fixes this.

-> This may or may not be related to the `gradient` install. If not, fix in container setup.

## Improvements

Various directions toward a full-stack QA testing suite. This would be project by someone QA-oriented, in collaboration with Product and Engineering, because the internal details of how to do QA and how Gradient works internally dictate the details of various tests/approaches.

 - The scripts just dump out whatever output; it could be formatted more readably
 - Likewise add comments what lines are doing if these are to be used going forward
 - Make them invokable from CLI to run whole test suite as a script: CLI can create project, notebook, run scripts, etc.
 - Generalize scripts to test all instance types, not single chosen type
 - Test other software libraries that are not directly used by users but nevertheless have required version ranges crucial to Gradient Notebooks/Workflows/Deployments functioning correctly. E.g., we test CUDA indirectly by running models, but not directly. Maybe PyYAML?
 - Run containers with the Container -> Command field under Advanced Options populated. Currently this is blank but notebook/lab works in Jupyter interface. [5]
 - The corresponding Git repos to the TF and PyTorch containers, https://github.com/gradient-ai/TF2.4.1.git and https://github.com/gradient-ai/PyTorch-1.8.git should be updated and named appropriately, e.g., use the TF versions corresponding to the containers. These are also not currently used when running the above scripts.

[4] Current commands are

 - TF: `jupyter notebook --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.trust_xheaders=True --NotebookApp.disable_check_xsrf=False --NotebookApp.allow_remote_access=True --NotebookApp.allow_origin='*'`
 - PyTorch: `jupyter lab --allow-root --ip=0.0.0.0 --no-browser --LabApp.trust_xheaders=True --LabApp.disable_check_xsrf=False --LabApp.allow_remote_access=True --LabApp.allow_origin='*'`

Linear https://linear.app/paperspace/issue/NB-36/support-jupyter-v6 says if we run Jupyter V6 the command is updated to

`jupyter-notebook --allow-root --ServerApp.allow_root=True --ServerApp.ip=0.0.0.0 --no-browser --ServerApp.trust_xheaders=True --ServerApp.disable_check_xsrf=False --ServerApp.allow_remote_access=True --ServerApp.allow_origin='*'`

Ideally we should run JupyterLab 3.x on all containers rather than just Jupyter notebook.
