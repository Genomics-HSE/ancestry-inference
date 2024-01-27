FROM cschranz/gpu-jupyter:v1.6_cuda-12.0_ubuntu-22.04

RUN pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
RUN pip install torch-geometric
