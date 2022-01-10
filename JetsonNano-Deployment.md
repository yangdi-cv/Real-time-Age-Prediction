# Deploy the real-time demo on Jetson Nano 2GB

## 1. Jetson Nano Image Writing
1. Image File Download </br>
Select and download the image file of Jetson Nano 2GB version from NVIDIA official website: https://developer.nvidia.com/embedded/downloads </br>
<img src="https://github.com/Ericdiii/Real-time-Age-Prediction/blob/demo/image/1.png?raw=true" height="200"/>

2. Image Writing </br>
Etcher is a tool for writing the image to the SD card. We can download it from: https://www.balena.io/etcher/ </br>
<img src="https://github.com/Ericdiii/Real-time-Age-Prediction/blob/demo/image/2.png?raw=true" height="120"/>

## 2. Deeper Learning Framework Construction

1. Python Environment Configuration (in the Jetson Nano Terminal) </br>
```
- sudo apt-get update
- sudo apt-get upgrade
- sudo apt-get install git cmake python3-dev
- sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev
- sudo apt-get install python3-pip
- sudo pip3 install -U pip testresources setuptools
- echo alias python=python3 >> ~/.bashrc
- source ~/.bashrc
```
2. PyTorch Framework Configuration </br>
```
- sudo apt-get install libopenblas-base libopenmpi-dev
- sudo pip3 install mpi4py
- sudo pip3 install Cython
```
- **Install torch 1.6.0** </br>
Download the torch 1.6.0 whl file: https://nvidia.box.com/shared/static/9eptse6jyly1ggt9axbja2yrmj6pbarc.whl </br>
Type the following command in the download file path
```
- sudo pip3 install torch-1.6.0-cp36-cp36m-linux_aarch64.whl
```
- **Install torchvision 0.7.0** </br>
Download the torchvision 0.7.0: https://github.com/pytorch/vision/tree/release/0.7 </br>
Unzip the package, and type the following command in the path of “setup.py”
```
- sudo apt install libavcodec-dev
- sudo apt install libavformat-dev
- sudo apt install libswscale-dev
- sudo python3 setup.py install
```
## 3. Real-time System Implementation
1. Download the code and pre-trained model from: https://github.com/Ericdiii/Real-time-Age-Prediction </br>
This project applies multiple third-party libraries, including `numpy`, `imutils`, `time`, `cv2`, `PIL`, `torch`, and `torchvision`. </br>
The libraries of numpy, time, cv2, and PIL have already been installed in the official Jetson Nano image file. </br>
Therefore, we only need to install `imutils` additionally:
```
- sudo pip install imutils
```
2. Demonstration (webcam)
```
- python AP_System.py
```
