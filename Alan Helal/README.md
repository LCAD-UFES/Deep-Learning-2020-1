<h1 align="center">
    <a href="https://pt-br.reactjs.org/">🔗 Using Convolutional Neural Network for Animal Classification</a>
</h1>
<p align="center">🚀 Step by step of how to use my project</p>

This project aims to solve a real-life problem. Trap cameras are palced around farms capturing images of local wildlife. They are activated when they detect movement, however they are often activated due to wind, tree movement or even falling fruit. The idea is to identify in the photos taken by trap cameras three types of animals: armadillo, deer and wild pig.
For training and validation of the neural network a dataset with more than 5000 photos taken by trap cameras was used. Training requires computing power hardly found in personal computers in our homes. To solve this problem, Google Colab was used for training and validation.

Next, it will be explained how to use the created neural network to solve the classification problem. For your convenience, two models have been made available on Google Colab:
* The untrained neural network. It will be necessary to download the entire dataset (3.81 GB) and train the network before using it.
* The neural network trained and ready to receive the photos you want to classify.

For more technical information regarding the project, please read the article Use of Convolutional Neural Networks for Wild Animal Classification available on this GitHub.
If you wish to execute the pre-trained neural network, please [click here.](#trained-neural-network)

# Untrained Neural Network

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
The last Debug Step to check if the training dataset is correct is to print the tensor shape. Accordingly to our previously configuration, the tensor shpae should be (32, 224, 224, 3) where 32 represents the batch size, 224 the image height and width and 3 refers to the RGB channels of the image. We are also printing the label batch shape we should get (32,) since we have 32 labels (one for each image in the batch).

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

#### Step 4

We are dealing with a huge amount of data, we need to configure the dataset for performance. We'll be using a buffered prefetching to avoid I/O blocking when loading data. The cache() method keeps the images in memory after they're loaded off disk during the fisrt epoch of training. This will ensure the dataset does not become a bottleneck while training the model. Since the dataset is too large to fit into memory, this method is used to create a performant on-disk cache. The prefetch() method overlaps data preprocessing and model execution while training.

The code used is:
```python
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

There is no output for this step.

#### Step 5

Now it is time to create our model using VGG16. Since we didnt standardize the data before, we need to add a normalization layer in the model to change the range of the RGB values from 0 to 255 to be standarzie values in the 0-1 range.

The VGG model with the normalization layer is created by the code below:

```python
num_classes = 3 #Veado, Porco, Tatu

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  #Primeiro bloco convolucional
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  
  #Segundo bloco convolucional
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  #Terceiro bloco convolucional
  layers.Conv2D(256, 3, padding='same', activation='relu'),
  layers.Conv2D(256, 3, padding='same', activation='relu'),
  layers.Conv2D(256, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  #Quarto bloco convolucional
  layers.Conv2D(512, 3, padding='same', activation='relu'),
  layers.Conv2D(512, 3, padding='same', activation='relu'),
  layers.Conv2D(512, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  #Quinto bloco convolucional
  layers.Conv2D(512, 3, padding='same', activation='relu'),
  layers.Conv2D(512, 3, padding='same', activation='relu'),
  layers.Conv2D(512, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  #Camadas Densas (Fully Connected)
  layers.Flatten(),
  layers.Dense(4096, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(4096, activation='relu'),
  layers.Dense(num_classes, activation = 'softmax')
])


model.summary()
```

Since we are printing the model summary, you should see the following output:

```bash
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
rescaling_2 (Rescaling)      (None, 224, 224, 3)       0         
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 224, 224, 64)      1792      
_________________________________________________________________
conv2d_14 (Conv2D)           (None, 224, 224, 64)      36928     
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 112, 112, 64)      0         
_________________________________________________________________
conv2d_15 (Conv2D)           (None, 112, 112, 128)     73856     
_________________________________________________________________
conv2d_16 (Conv2D)           (None, 112, 112, 128)     147584    
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 56, 56, 128)       0         
_________________________________________________________________
conv2d_17 (Conv2D)           (None, 56, 56, 256)       295168    
_________________________________________________________________
conv2d_18 (Conv2D)           (None, 56, 56, 256)       590080    
_________________________________________________________________
conv2d_19 (Conv2D)           (None, 56, 56, 256)       590080    
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 28, 28, 256)       0         
_________________________________________________________________
conv2d_20 (Conv2D)           (None, 28, 28, 512)       1180160   
_________________________________________________________________
conv2d_21 (Conv2D)           (None, 28, 28, 512)       2359808   
_________________________________________________________________
conv2d_22 (Conv2D)           (None, 28, 28, 512)       2359808   
_________________________________________________________________
max_pooling2d_8 (MaxPooling2 (None, 14, 14, 512)       0         
_________________________________________________________________
conv2d_23 (Conv2D)           (None, 14, 14, 512)       2359808   
_________________________________________________________________
conv2d_24 (Conv2D)           (None, 14, 14, 512)       2359808   
_________________________________________________________________
conv2d_25 (Conv2D)           (None, 14, 14, 512)       2359808   
_________________________________________________________________
max_pooling2d_9 (MaxPooling2 (None, 7, 7, 512)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 25088)             0         
_________________________________________________________________
dense_3 (Dense)              (None, 4096)              102764544 
_________________________________________________________________
dropout_1 (Dropout)          (None, 4096)              0         
_________________________________________________________________
dense_4 (Dense)              (None, 4096)              16781312  
_________________________________________________________________
dense_5 (Dense)              (None, 3)                 12291     
=================================================================
Total params: 134,272,835
Trainable params: 134,272,835
Non-trainable params: 0
```

#### Debug Step 4

If you wish to see a SVG representation of the model just created, execute this debug step.

#### Step 6

Since the photos from the dataset were taken in the wild, the use of Data Augmentation would help the training because the dataset contains several photos of the same animal in different angles and positions.

The code used for Data Augmentation is:

```python
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")
```

You should see an output of 9 horizontaly rotated images:

![image](https://user-images.githubusercontent.com/19311371/121207189-77094580-c84f-11eb-8d26-3072b4555bac.png)

#### Step 7

The last step beofre the traingin is to compile the model and select the optmizer, the loss function and the metrics. We are using Adam as optmizer, Sparse Categorical Cross Entropy as the loss function and our metric is accuracy. 

The code is:

```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

#### Step 8

Now we have everything ready to train our neural network. Just run the code below to use 10 epochs of training.

```python
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
```

The output should be something like this:

```bash
Epoch 1/10
114/114 [==============================] - 165s 1s/step - loss: 6.8301 - accuracy: 0.5508 - val_loss: 0.7702 - val_accuracy: 0.6484
Epoch 2/10
114/114 [==============================] - 98s 859ms/step - loss: 0.6504 - accuracy: 0.7093 - val_loss: 0.6587 - val_accuracy: 0.7044
Epoch 3/10
114/114 [==============================] - 98s 860ms/step - loss: 0.5781 - accuracy: 0.7618 - val_loss: 0.6000 - val_accuracy: 0.7275
Epoch 4/10
114/114 [==============================] - 97s 851ms/step - loss: 0.5120 - accuracy: 0.7970 - val_loss: 0.5652 - val_accuracy: 0.7725
Epoch 5/10
114/114 [==============================] - 97s 853ms/step - loss: 0.5033 - accuracy: 0.8093 - val_loss: 0.4477 - val_accuracy: 0.8374
Epoch 6/10
114/114 [==============================] - 97s 852ms/step - loss: 0.3603 - accuracy: 0.8673 - val_loss: 0.3972 - val_accuracy: 0.8670
Epoch 7/10
114/114 [==============================] - 97s 854ms/step - loss: 0.3073 - accuracy: 0.8882 - val_loss: 0.3081 - val_accuracy: 0.8846
Epoch 8/10
114/114 [==============================] - 97s 852ms/step - loss: 0.4469 - accuracy: 0.8346 - val_loss: 0.3935 - val_accuracy: 0.8505
Epoch 9/10
114/114 [==============================] - 97s 849ms/step - loss: 0.3071 - accuracy: 0.8865 - val_loss: 0.3275 - val_accuracy: 0.8868
Epoch 10/10
114/114 [==============================] - 97s 851ms/step - loss: 0.2200 - accuracy: 0.9280 - val_loss: 0.2475 - val_accuracy: 0.9165
```
Note that after the training the neural network has an accuracy of 92.80%.

### Optional Step

If you wish to save the entire trained network, just run this step and then download the meuModeloTreinadoVgg16.h5 file generated. You can import the network later and use it without having to train it again.

#### Debug Step 5

We can plot the Training and Validation Accuracy and Training and Validation Loss graphs after the training. Just run the code below to see the graphs:

```python
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```

The output should be:

![image](https://user-images.githubusercontent.com/19311371/121209366-2bf03200-c851-11eb-92f6-c90e9866255a.png)

#### Step 9

Now we can use our test dataset to test the neural network. First we are going to use only the armadillo photos and see how many of them the neural network classify correctly.

After runing the code below: 

```python
# Testando apenas com fotos de Tatu

import matplotlib.image as mpimg
tatus = pathlib.Path('/content/datasetFlorestaV3/teste/tatu')
images = list(tatus.glob('*.JPG'))

v = 0
p = 0
t = 0

for i in images:
    img = mpimg.imread(i)
    img = keras.preprocessing.image.load_img(
        i, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    if class_names[np.argmax(score)] == "porco":
        p = p + 1
    elif class_names[np.argmax(score)] == "tatu":
        t = t + 1
    else:
        v = v + 1
  
print('Veado: {}'.format(v)) 
print('Porco: {}'.format(p)) 
print('Tatu: {}'.format(t)) 
total = v + p + t
acerto =(t/total)*100
print('A rede acertou {:.2}% das imagens' .format(acerto))
```

You should see the following output:

```bash
Veado: 2
Porco: 0
Tatu: 246
A rede acertou 99.19% das imagens
```

Then we can test for wild pig photos:

```python
# Testando apenas com fotos de Porcos
porcos = pathlib.Path('/content/datasetFlorestaV3/teste/porco')
images = list(porcos.glob('*.JPG'))

v = 0
p = 0
t = 0

for i in images:
    img = mpimg.imread(i)
    img = keras.preprocessing.image.load_img(
        i, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    if class_names[np.argmax(score)] == "porco":
        p = p + 1
    elif class_names[np.argmax(score)] == "tatu":
        t = t + 1
    else:
        c = v + 1


print('Porco: {}'.format(p)) 
print('Tatu: {}'.format(t)) 
print('Veado: {}'.format(v)) 
total = v + p + t
acerto =(p/total)*100
print('A rede acertou {}% das imagens' .format(acerto))
```
The output should be:

```python
Porco: 164
Tatu: 24
Veado: 0
A rede acertou 87.23% das imagens
```

And the last one would be the deer:

```python
# Testando apenas com fotos de Veados
veados = pathlib.Path('/content/datasetFlorestaV3/teste/veado')
images = list(veados.glob('*.JPG'))


v = 0
p = 0
t = 0

for i in images:
    img = mpimg.imread(i)
    img = keras.preprocessing.image.load_img(
        i, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    if class_names[np.argmax(score)] == "porco":
        p = p + 1
    elif class_names[np.argmax(score)] == "tatu":
        t = t + 1
    else:
        v = v + 1 

print('Porco: {}'.format(p)) 
print('Tatu: {}'.format(t)) 
print('Veado: {}'.format(v)) 
total = v + p + t
acerto =(v/total)*100
print('A rede acertou {:.2}% das imagens' .format(acerto))
```

```bash
Porco: 68
Tatu: 4
Veado: 82
A rede acertou 53.24% das imagens
```

#### Optional Step 2

If you wish to see how the neural network is classifying each image you can execute this step for each class. The output should be the image with the prediction and score given by the neural netowrk. An example of output is showed below:

<img width="359" alt="Schermata 2021-06-08 alle 12 19 59" src="https://user-images.githubusercontent.com/19311371/121212589-e2551680-c853-11eb-85ef-c4deaea30961.png">


# Trained Neural Network

Why train something already trained? For your convenience you can test the neural network without having to download almost 4 GB of data and spend time training it. Just [click here](https://colab.research.google.com/drive/1AZE2scb09Eixsfgydl-lnvbfuaTLb3uU?hl=en#scrollTo=pz30YZNqu4op) to open the Trained Neural Network. It has only 5 steps.

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
We need to download the dataset containing the images for training, validation and testing. After downloading, we decompress the file and delete every hidden file that might be created during the compressing. This step might take a while depending on yout internet speed. After its done you should see the number 615 as an output. It indicates the number of images in our dataset.

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

#### Step 3

We need to download the Trained Neural Network. The code below will do it for you:

```python
!gdown --id 1Nr29d6b8Picr_LqqYb3YN416E50fShzq
```

#### Step 4

We need to create the model from the file we just downloaded. Again, just execute the code below and after a few seconds you will have the entire network trained at your service.

```python
from keras.models import load_model
model = load_model('/content/CNNTreinadaVGG16.h5')
```

#### Step 5

Now we can use our test dataset to test the neural network. First we are going to use only the armadillo photos and see how many of them the neural network classify correctly.

After runing the code below: 

```python
# Testando apenas com fotos de Tatu

import matplotlib.image as mpimg
tatus = pathlib.Path('/content/datasetFlorestaV3/teste/tatu')
images = list(tatus.glob('*.JPG'))

v = 0
p = 0
t = 0

for i in images:
    img = mpimg.imread(i)
    img = keras.preprocessing.image.load_img(
        i, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    if class_names[np.argmax(score)] == "porco":
        p = p + 1
    elif class_names[np.argmax(score)] == "tatu":
        t = t + 1
    else:
        v = v + 1
  
print('Veado: {}'.format(v)) 
print('Porco: {}'.format(p)) 
print('Tatu: {}'.format(t)) 
total = v + p + t
acerto =(t/total)*100
print('A rede acertou {:.2}% das imagens' .format(acerto))
```

You should see the following output:

```bash
Veado: 0
Porco: 7
Tatu: 241
A rede acertou 97.18% das imagens
```

Then we can test for wild pig photos:

```python
# Testando apenas com fotos de Porcos
porcos = pathlib.Path('/content/datasetFlorestaV3/teste/porco')
images = list(porcos.glob('*.JPG'))

v = 0
p = 0
t = 0

for i in images:
    img = mpimg.imread(i)
    img = keras.preprocessing.image.load_img(
        i, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    if class_names[np.argmax(score)] == "porco":
        p = p + 1
    elif class_names[np.argmax(score)] == "tatu":
        t = t + 1
    else:
        c = v + 1


print('Porco: {}'.format(p)) 
print('Tatu: {}'.format(t)) 
print('Veado: {}'.format(v)) 
total = v + p + t
acerto =(p/total)*100
print('A rede acertou {}% das imagens' .format(acerto))
```
The output should be:

```python
Porco: 184
Tatu: 0
Veado: 0
A rede acertou 100.00% das imagens
```

And the last one would be the deer:

```python
# Testando apenas com fotos de Veados
veados = pathlib.Path('/content/datasetFlorestaV3/teste/veado')
images = list(veados.glob('*.JPG'))


v = 0
p = 0
t = 0

for i in images:
    img = mpimg.imread(i)
    img = keras.preprocessing.image.load_img(
        i, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    if class_names[np.argmax(score)] == "porco":
        p = p + 1
    elif class_names[np.argmax(score)] == "tatu":
        t = t + 1
    else:
        v = v + 1 

print('Porco: {}'.format(p)) 
print('Tatu: {}'.format(t)) 
print('Veado: {}'.format(v)) 
total = v + p + t
acerto =(v/total)*100
print('A rede acertou {:.2}% das imagens' .format(acerto))
```

```bash
Porco: 14
Tatu: 14
Veado: 126
A rede acertou 81.82% das imagens
```

#### Optional Step 

If you wish to see how the neural network is classifying each image you can execute this step for each class. The output should be the image with the prediction and score given by the neural netowrk.
