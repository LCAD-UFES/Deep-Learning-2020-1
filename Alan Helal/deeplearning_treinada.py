# -*- coding: utf-8 -*-
"""DeepLearning_Treinada.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AZE2scb09Eixsfgydl-lnvbfuaTLb3uU

# Passo 1

*   Importando as bibliotecas do TensorFlow e Keras
*   Importando as bibliotecas do numpy e PIL
*   Verificando se está utilizando GPU para o treinamento
"""

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

"""# Passo 2


*   Efetuando o download do dataset de treinamento e teste (454 MB)
*   Descompactando o dataset
*   Ajustando o dataset para eliminar arquivos temporários e ocultos que possam ter sido criados no processo de compactação/descompactação
*   Contando o número de imagens JPG para verificar se coincide com a quantidade de imagens do dataset (615 imagens divididas em 3 categorias: Veado, Tatu e Porco)

"""

import pathlib

!gdown --id 1mxKdq2ec56rLKkS-9BA1lWz9WsEOec5A #Efetuando o download do dataset direto do meu Google Drive
!tar -xzvf /content/datasetFlorestaTESTE.tgz #Descompactando o dataset
data_dir = pathlib.Path('/content/datasetFlorestaV3/teste') #Definindo o diretório de treinamento

#Os comandos abaixo servem para remover todos os arquivos ocultos/temporários das pastas do dataset
!rm -rf /content/datasetFlorestaV3/teste/porco/._*
!rm -rf /content/datasetFlorestaV3/teste/tatu/._*
!rm -rf /content/datasetFlorestaV3/teste/veado/._*

image_count = len(list(data_dir.glob('*/*.JPG'))) #Contando quantas imagens possuem o dataset de treinamento
print(image_count) #verificar se o número de imagens importadas coincide com o número de imagens do dataset de treinamento

"""# Passo 3

Realizando o download da rede treinada
"""

!gdown --id 1Nr29d6b8Picr_LqqYb3YN416E50fShzq

"""# Passo 4

Recriando o modelo da rede neural utilizando VGG16 já com a arquitetura e os pesos.
"""

from keras.models import load_model
model = load_model('/content/CNNTreinadaVGG16.h5')

"""# Passo 5

Testando com a base de teste conhecida


*   Testando apenas os Tatus para ver quantos a rede acerta
*   Testando apenas os Porcos para ver quantos a rede acerta
*   Testando apenas os Veados para ver quantos a rede acerta

"""

# Testando apenas com fotos de Tatu
class_names = ['porco', 'tatu', 'veado']
import matplotlib.image as mpimg
tatus = pathlib.Path('/content/datasetFlorestaV3/teste/tatu')
images = list(tatus.glob('*.JPG'))

v = 0
p = 0
t = 0

for i in images:
    img = mpimg.imread(i)
    img = keras.preprocessing.image.load_img(
        i, target_size=(400, 400)
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
print('A rede acertou {:.2f}% das imagens' .format(acerto))

# Testando apenas com fotos de Porcos
porcos = pathlib.Path('/content/datasetFlorestaV3/teste/porco')
images = list(porcos.glob('*.JPG'))

v = 0
p = 0
t = 0

for i in images:
    img = mpimg.imread(i)
    img = keras.preprocessing.image.load_img(
        i, target_size=(400, 400)
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
print('A rede acertou {:.2f}% das imagens' .format(acerto))

# Testando apenas com fotos de Veados
veados = pathlib.Path('/content/datasetFlorestaV3/teste/veado')
images = list(veados.glob('*.JPG'))


v = 0
p = 0
t = 0

for i in images:
    img = mpimg.imread(i)
    img = keras.preprocessing.image.load_img(
        i, target_size=(400, 400)
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
print('A rede acertou {:.2f}% das imagens' .format(acerto))

"""# Passo opcional

Classificação com score
Vendo como cada foto do dataset de teste foi classificada com seu respectivo score.
"""

# Commented out IPython magic to ensure Python compatibility.
# Apenas fotos de tatu
# %pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

veados = pathlib.Path('/content/datasetFlorestaV3/teste/tatu')
images = list(veados.glob('*.JPG'))
for i in images:
    img = mpimg.imread(i)
    imgplot = plt.imshow(img)
    plt.show()
    img = keras.preprocessing.image.load_img(
        i, target_size=(400, 400)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print("Esse imagem é: \033[1m{}!\033[0m Com \033[1m{:.2f}%\033[0m de certeza.".format(class_names[np.argmax(score)], 100 * np.max(score)))

# Commented out IPython magic to ensure Python compatibility.
# Apenas fotos de porco
# %pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

veados = pathlib.Path('/content/datasetFlorestaV3/teste/porco')
images = list(veados.glob('*.JPG'))
for i in images:
    img = mpimg.imread(i)
    imgplot = plt.imshow(img)
    plt.show()
    img = keras.preprocessing.image.load_img(
        i, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print("Esse imagem é: \033[1m{}!\033[0m Com \033[1m{:.2f}%\033[0m de certeza.".format(class_names[np.argmax(score)], 100 * np.max(score)))

# Commented out IPython magic to ensure Python compatibility.
# Apenas fotos de Veado
# %pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

veados = pathlib.Path('/content/datasetFlorestaV3/teste/veado')
images = list(veados.glob('*.JPG'))
for i in images:
    img = mpimg.imread(i)
    imgplot = plt.imshow(img)
    plt.show()
    img = keras.preprocessing.image.load_img(
        i, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print("Esse imagem é: \033[1m{}!\033[0m Com \033[1m{:.2f}%\033[0m de certeza.".format(class_names[np.argmax(score)], 100 * np.max(score)))