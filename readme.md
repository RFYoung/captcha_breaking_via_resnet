# Recognition of Text Verification Code Based on Residual Network

In this project, a VGG-Like model, a ResNet-34-Like model and a ResNet-50-Like model are used to break an enhanced captcha from the ["captcha" project](https://github.com/lepture/captcha/ ).

## Requirements:

To run this project, you need a computer with Nvidia GPU that support CUDA, and runs the Linux operating system. 

PyTorch and Torchvision witb CUDA support is the main library used in this project. Pillow, Numpy and Scipy are also used to generate captcha images. Tensorboard is also used to view and export the result of loss and accuracy of the project.

To install all the libraries except Torchvision on Archlinux or Manjaro, run the following command in terminal:

```bash
sudo pacman -Syu python-pytorch-cuda python-scipy python-numpy python-pillow tensorboard
```

You can install Torchvision via AUR is possible, but to use a Python virtual environment is recommended. To set a blank virtual environment, run the following command:

```bash
python -m venv venv --system-site-packages
```

To activate the virtual environment,  we could switch the project directory and use the following command

```bash
source ./venv/bin/activate  
```

Then we could install the Torchvision via PIP command:

```bash
pip install torchvision
```

## Run the code

Firstly, you need to  switch to the project directory, and then activate the virtual environment via the command:

```bash
source ./venv/bin/activate  
```

Before running the main function, there are small adjustments in  the `main.py` file. The file includes 3 models:

```python
model = CnnModel(CAPTCHA_CHAR_LEN, CAPTCHA_STR_LEN, input_shape=(3, PIC_HEIGHT, PIC_WIDTH))
model = ResNet50Model(CAPTCHA_CHAR_LEN,CAPTCHA_STR_LEN, input_shape=(3, PIC_HEIGHT, PIC_WIDTH))
model = ResNet34Model(CAPTCHA_CHAR_LEN,CAPTCHA_STR_LEN, input_shape=(3, PIC_HEIGHT, PIC_WIDTH))
```

You need to choose one model and comment out other two. Then we need to change where to store the training results, we need to change the following variable to the directory to store the results:

```
STORE_DIRECTORY = "log/model_1/"
```

Then we could start up the main project via the following command:

```
python main.py
```

It takes some hours to run the programme, then we could run the command to launch tensorboard and view the result:

```
tendorboard --logdir ./
```

