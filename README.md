# EASYContrastiveLearning

Basic framework intended to test in a controlled environment multimodal contrastive learning functions.

Multimodal MNIST: MNIST+Audio_MNIST

Involves three modalities: Textual labels (from "zero" to "nine"), images from MNIST and audio from Audio_MNIST.

## How to clone:

git clone https://github.com/Giordano-Cicchetti/EasyCL.git

cd EasyCL


## How to install dependencies

conda create -n EasyCL python=3.9

conda activate EasyCL

pip install -r requirements.txt

## How to download dataset:

mkdir mnist_data

MNIST normale forse lo scarica direttamente il codice alla prima run

Download audioMNIST from kaggle https://www.kaggle.com/datasets/sripaadsrinivasan/audio-mnist/data

Once downloaded, extract it and put in the folder mnist_data/audio-MNIST

Structure of the mnist_data folder:

--mnist_data

---- audio-MNIST

-------- data

------------- 01

------------- 02

------------- 03

------------- etc...

---- MNIST

-------- raw

------------- t10k-images-idx3-ubyte

------------- etc...



## Download Word2Vec

GoogleNews-vectors-negative300.bin from kaggle https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300

Extract and put only the GoogleNews-vectors-negative300.bin in this folder


## How to run

conda activate EasyCL

python ./run.py

## How to act

**dataloader.py** is devoted to the data loading. Create dataset and dataloaders for Multimodal MNIST. Touch it if you want to change dataset/add modality/correct some mistak/do something strange!

**loss.py** contains the loss functions

**pipelines.py** contains training and testing pipelins along with visualization functions. Maybe visualization funtions should be moved to another py file.

**models.py** contains encoders. At the moment we have Word2Vec as textual encoder. Two simple CNNs as visual and audio encoders.

**run.py** set all variables and hyperparameters, initialize dataset/dataloaders and run a pipeline.



