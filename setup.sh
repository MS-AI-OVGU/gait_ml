conda create -y --name spine python==3.11.13
conda activate spine

conda install -y -c conda-forge pandas=2.2.3
conda install -y -c conda-forge tabulate=0.9.0
conda install -y -c conda-forge matplotlib=1.15.2
conda install -y -c conda-forge numpy=2.2.5
conda install -y -c conda-forge scipy=1.15.2
conda install -y -c conda-forge filterpy=1.4.5
conda install -y -c conda-forge scikit-learn=1.6.1
conda install -y -c conda-forge xlrd=2.0.1
conda install -y -c conda-forge openpyxl=3.1.5
conda install -y -c conda-forge seaborn=0.13.2
conda install -y -c conda-forge numba=0.61.2
conda install -y -c conda-forge tensorboard=2.19.0
conda install -y -c conda-forge py-xgboost=2.1.1
conda install -y -c conda-forge plotly=6.2.0
conda install -y -c conda-forge grpcio=1.54.2 grpcio-tools=1.54.2 libgrpc=1.54.2 absl-py
conda install -y -c conda-forge pingouin=0.5.5
conda install -y -c conda-forge nbformat=5.10.4
conda install -y -c conda-forge ipykernel=6.29.4
python -m ipykernel install --user --name spine --display-name="spine"

#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install gaitmap==2.5.2
pip install gaitmap_mad==2.5.2
pip install ahrs==0.4.0 
pip install sentence-transformers==3.0.1
pip install faiss-cpu==1.7.4
pip install huggingface_hub==0.36.0
pip install transformers==4.57.1