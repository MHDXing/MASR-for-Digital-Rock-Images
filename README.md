# Multi Attention Super-Resolution Neural Network (MASR)

This repository is an official PyTorch implementation of the paper **"Digital rock resolution enhancement and detail recovery with multi attention neural network
"**

The source code is primarily derived from [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch) and [CDCSR](https://github.com/xiezw5/Component-Divide-and-Conquer-for-Real-World-Image-Super-Resolution).
We provide full training and testing codes, pre-trained models used in our paper. You can train your model from scratch, or use a pre-trained model to enlarge your digital rock images.



## Code
### Dependencies
* Python 3.8.5
* PyTorch = 1.8.1
* numpy
* cv2
* skimage
* tqdm

### Quick Start

```bash
git clone https://github.com/MHDXing/MASR-for-Digital-Rock-Images.git
cd MASR-for-Digital-Rock-Images-main/MASR
```

## Dataset
The dataset we used was derived from [DeepRockSR](https://digitalrocks-dev.tacc.utexas.edu/media/projects/215/archive.zip).
There are 9600, 1200, 1200 HR 2D images (500x500) for training, testing and validation, respectively.

#### Training
1. Download the dataset and unpack them to any place you want. Then, change the ```dataroot``` and ```test_dataroot``` argument in ```./options/realSR_MASR.py``` to the place where images are located
2. You can change the hyperparameters of different models by modifying the files in the ```./options/``` folder
3. Run ```CDC_train_test.py``` using script file ```train_pc.sh```
```bash
bash train_pc.sh
```
4. You can find the results in ```./experiments/MASR_x4``` if the ```exp_name``` argument in ```./options/realSR_MASR.py``` is ```MASR_x4```.

#### Testing
1. Download our pre-trained models to ```./models``` folder or use your pre-trained models
2. Change the ```test_dataroot``` argument in ```CDC_test.py``` to the place where images are located
3. Run ```CDC_test.py``` using script file ```test_models_pc.sh```
```bash
bash test_models_pc.sh
```
4. You can find the enlarged images in ```./results``` folder.

### Pretrained models
[MASR Models](https://drive.google.com/file/d/1N2WcFEchQbUNNB6PbmpMMsTOor4nt0jr/view?usp=sharing)
<!-- 1. [MASR Models](https://drive.google.com/file/d/18Bg1B5XvksMNsM1KXoPegsOhIbP6WnC4/view?usp=sharing)
2. [EDSR Models](https://drive.google.com/file/d/1GGcnUCGaBWStxh-78PnDIlaCfPMfATaG/view?usp=sharing)
3. [RCAN Models](https://drive.google.com/file/d/1VhppmVr159dlXzbVPh0zDcVGBblj2k0j/view?usp=sharing)
4. [CDCSR Models](https://drive.google.com/file/d/18Bg1B5XvksMNsM1KXoPegsOhIbP6WnC4/view?usp=sharing) -->



