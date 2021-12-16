pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -Uq "tfp-nightly[jax]" 
conda install -c conda-forge gh
# jupyterlab things
conda install -c conda-forge nodejs 
pip install --upgrade jupyterlab-git
jupyter labextension install base16-mexico-light
conda install -c conda-forge nodejs
conda update nodejs
# Folders
mkdir documents/
