## Installation
### 1. Installing on the host machine
Step1. Install adEye.
```shell
conda create -n adEye python==3.8
activate adEye
mkdir adEye
cd adEye
conda install git

step2. CUDA, cuDNN, pytorch Install.

Step3.Visual Studio Installer C++ install

Step4. Install adEye requirements.txt.
```shell
git clone https://github.com/2022-SMHRD-IS-AI3/ad-eye-python.git
pip install -r requirements.txt
python setup.py develop
