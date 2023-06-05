## Installation
### 1. Installing on the host machine
Step1. Install adEye.
```shell
conda create -n adEye python==3.8
activate adEye
mkdir adEye
cd adEye
conda install git
git clone https://github.com/2022-SMHRD-IS-AI3/ad-eye-python.git
pip install -r requirements.txt
python setup.py develop
