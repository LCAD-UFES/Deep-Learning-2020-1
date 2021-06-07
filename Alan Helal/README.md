<h1 align="center">
    <a href="https://pt-br.reactjs.org/">🔗 Using Convolutional Neural Network for Animal Classification</a>
</h1>
<p align="center">🚀 Step by step of how to use my project</p>

This project aims to solve a real-life problem. Trap cameras are palced around farms capturing images of local wildlife. They are activated when they detect movement, however they are often activated due to wind, tree movement or even falling fruit. The idea is to identify in the photos taken by trap cameras three types of animals: armadillo, deer and wild pig.
For training and validation of the neural network a dataset with more than 5000 photos taken by trap cameras was used. raining requires computing power hardly found in personal computers in our homes. To solve this problem, Google Colab was used for training and validation.

Next, it will be explained how to use the created neural network to solve the classification problem. For your convenience, two models have been made available on Google Colab:
* The untrained neural network. It will be necessary to download the entire dataset (3.81 GB) and train the network before using it.
* The neural network trained and ready to receive the photos you want to classify.

For more technical information regarding the project, please read the article Use of Convolutional Neural Networks for Image Classification available on this GitHub.
If you wish to execute the pre-trained neural network, please [click here.](#PretrainedNeuralNetwork)

<h2>Untrained Neural Network</h2>

Since we are using the Google Colab plataform, there is no need to prepare an environment to execute the Neural Network. Simply [click here](https://colab.research.google.com/drive/13nPasXwH141iL9O5fqDcbtoYr5VVhpv7?hl=en) to open the Google Colab Notebok containing the untrained neural network.
First thing we need to do is change the runtime type to use a GPU as hardware accelerator. To do so, select Runtime and then Select Runtime Type. Under Hardware Accelerator, choose GPU in the dropdown menu and then click in save. That's all the configuration we need to do.

Note: to execute every step just click on the play button located at the top left corner of each box that contain code.

#### Step 1
We need to import all the libraries used to train and validate the neural network (numpy, tensorflow and keras). Also, we need to import the libraries used to manipulate the files and plot graphs (matlibplot, os and PIL). After executing Step 1 you should get an output like this:

```json
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 11233807447978002051
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 11344216064
locality {
  bus_id: 1
  links {
  }
}
incarnation: 8972007624555681685
physical_device_desc: "device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7"
]
```
If you are using the correct runtime type, you shoudl see a GPU listed as a device_type.

The code used to do all the imports is:

```python
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
tf.config.list_physical_devices('GPU')
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices()) #imprimindo os tipos de dispositivos
```

#### Step 2
We need to download the dataset containing the images for training, validation and testing. After downloading, we decompress the file and delete every hidden file that might be created during the compressing. This step might take a while depending on yout internet speed. After its done you should see the number 4550 as an output. It indicates the number of images in our dataset.

The code used is:

```python
import pathlib

!gdown --id 1j64rDwoltrcbYiVaAEI-AZTaRjISP2tv #Efetuando o download do dataset direto do meu Google Drive
!tar -xzvf /content/datasetFlorestaV3.tgz #Descompactando o dataset
data_dir = pathlib.Path('/content/datasetFlorestaV3/treinamento') #Definindo o diretório de treinamento

#Os comandos abaixo servem para remover todos os arquivos ocultos/temporários das pastas do dataset
!rm -rf /content/datasetFlorestaV3/treinamento/porco/._*
!rm -rf /content/datasetFlorestaV3/treinamento/tatu/._*
!rm -rf /content/datasetFlorestaV3/treinamento/veado/._*
!rm -rf /content/datasetFlorestaV3/teste/porco/._*
!rm -rf /content/datasetFlorestaV3/teste/tatu/._*
!rm -rf /content/datasetFlorestaV3/teste/veado/._*

image_count = len(list(data_dir.glob('*/*.JPG'))) #Contando quantas imagens possuem o dataset de treinamento
print(image_count) #verificar se o número de imagens importadas coincide com o número de imagens do dataset de treinamento
```

#### Debug Step 1

This is not a necessary step but i highly recommend you to execute it to be sure that everything went well during th download and decompress of the data set. As stated before, we have three classes: deer, armadillo and wild pig. If you see an aleatory image of each class after running this step you are good to go on.

An example of output for each class:
```python
veado = list(data_dir.glob('veado/*'))
PIL.Image.open(str(veado[200])) 
#Exemplo de imagem do dataset de treinamento - VEADO
```
![image](https://user-images.githubusercontent.com/19311371/121084214-c3ed0d80-c7b6-11eb-8a9e-6583a0e842d7.png)

```python
porco = list(data_dir.glob('porco/*'))
PIL.Image.open(str(porco[400])) 
#Exemplo de imagem do dataset de treinamento - PORCO
```
![image](https://user-images.githubusercontent.com/19311371/121084296-dcf5be80-c7b6-11eb-956a-9bbb0725d18c.png)

```python
tatu = list(data_dir.glob('tatu/*'))
PIL.Image.open(str(tatu[400])) 
#Exemplo de imagem do dataset de treinamento - TATU
```
![image](https://user-images.githubusercontent.com/19311371/121084350-f0088e80-c7b6-11eb-96b8-dbb4ba5058ef.png)


#### Step 3
We need to define the loader parameters to use a the batch size of 32 images and the image height and width to 224 pixels (default size for VGG). Then we configure the training dataset to use 80% of the training images for training and 20% for validation. This would result in using 3640 images for training the neural network and 910 images for validating the training.

The code used for the loader configuration is:
```python
#Para o treinamento serão utilizados batches de 32 imagens que serão redimensionadas para 400 por 400 pixels (altura e largura)
batch_size = 32
img_height = 224
img_width = 224
```

The code used to set the training dataset to use 80% of the images:
```python
#definindo que 80% das imagens do dataset será usada para treinamento
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
```
You should get the following output:
```bash
Found 4550 files belonging to 3 classes.
Using 3640 files for training.
```

The code used to set the validation dataset to use 20% of the images:
```python
#definindo que 20% das imagens do dataset será usada par validação
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
```
You should get the following output:
```bash
Found 4550 files belonging to 3 classes.
Using 910 files for training.
```

Just to check if the class names are correct, execute the code below and you should get as an output the name of the three classes:
```python
#verificando se foram criadas as classes com o nomes das pastas onde estão o dataset
class_names = train_ds.class_names
print(class_names)
```
```bash
['porco', 'tatu', 'veado']
```

#### Debug Step 2

To check if the training dataset was correctly configurated, run the code below to choose 9 aleatory images from the dataset and plot them with their classes. 
```python
#apenas para debug
#verificar se até o momento tudo que foi feito com o dataset está correto
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
```
If you get an output similar to this, it is safe to continue::

![image](https://user-images.githubusercontent.com/19311371/121085569-6c4fa180-c7b8-11eb-8d3b-5d1c59daaf56.png)

#### Debug Step 3
The last Debug Step to check if the training dataset is correct is to print the tensor shape. Accordingly to our previously configuration, the tensor shpae should be (32, 224, 224, 3) where 32 represents the batch size, 224 the image height and width and 3 refers to the RGB channels of the image. Since we are also printing the label batch shape we should get (32,) since we have 32 labels (one for eache image in the batch).

The code used is:
```python
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break
```

You should see the following output:
```bash
(32, 224, 224, 3)
(32,)
```
