# Test new container on Gradient CLI: nvcr.io/nvidia/pytorch:21.09-py3
#
# Last updated: Oct 20th 2021

# --- List packages on container ---

# Docker images can be complex structure (layers, mounted volumes, etc.), but basic package lists are available [1]

# Debian & Ubuntu-based containers
dpkg -l

# For ones installed by apt, conda, pip, etc.:
apt list --installed
conda env list
conda list -n base
pip3 list

# --- Check versions of packages of interest ---

# pip3 show <package name> can give more details on individual packages

apt --version
conda --version
git --version
gradient version
pip3 --version
jupyter --version # Returns core, notebook, ipython, and others

# --- Check GPU ---

nvidia-smi

# [1] https://stackoverflow.com/questions/57803595/how-do-i-list-all-applications-that-are-contained-in-a-docker-container/57804009
