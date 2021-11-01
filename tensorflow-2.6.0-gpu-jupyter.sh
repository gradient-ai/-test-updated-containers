# Test new container on Gradient CLI: tensorflow/tensorflow:2.6.0-gpu-jupyter
#
# Last updated: Oct 21st 2021

# --- List packages on container ---

# Docker images can be complex structure (layers, mounted volumes, etc.), but basic package lists are available [1]

# Debian & Ubuntu-based containers
dpkg -l

# For ones installed by apt, conda, pip, etc.:
apt list --installed
pip3 list

# --- Check versions of packages of interest ---

# pip3 show <package name> can give more details on individual packages

apt --version
git --version
gradient version
pip3 --version
jupyter --version # Returns core, notebook, ipython, and others

# --- Check GPU ---

nvidia-smi

# [1] https://stackoverflow.com/questions/57803595/how-do-i-list-all-applications-that-are-contained-in-a-docker-container/57804009
