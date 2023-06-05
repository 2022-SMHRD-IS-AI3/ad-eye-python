## Installation
### 1. Installing on the host machine
Step1. Install adEye.
```shell
conda create -n adEye python==3.8
activate adEye
cd adEye
git clone https://github.com/2022-SMHRD-IS-AI3/ad-eye-python.git
cd adEye
pip install -r requirements.txt
python setup.py develop
