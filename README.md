## Superresolution based on ESCPN

### Installation
```
pip3 install -r requirements.txt
```

### Usage

Train the model:
```
python3 train.py
```
Superresolution:
```
python3 rebuild.py
```
Evaluation:
```
python3 psnr.py
```

### Reference

Efficient Sub-Pixel Convolutional Neural Network(ESCPN)  
https://arxiv.org/pdf/1609.05158.pdf  
TensorFlow implementation of ESCPN  
https://github.com/kweisamx/TensorFlow-ESPCN  
https://github.com/Nyloner/TensorFlow-ESPCN  