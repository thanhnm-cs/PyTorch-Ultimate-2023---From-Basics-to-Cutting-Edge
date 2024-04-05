conda create -n pytorch python=3.10 -y
conda activate pytorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
conda install ipykernel -y
conda install -c anaconda seaborn -y
conda install scikit-learn -y
conda install -c conda-forge detecto -y
