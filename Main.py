import tensorflow as tf

from tensorflow import keras
from keras._tf_keras.keras import datasets, layers, models
import matplotlib.pyplot as plt
import  numpy as np
import seaborn as sns
import pandas as pd

print("Versão do TensorFlow:", tf.__version__)

logdir = 'log'

# Carregando os dados do CIFAR-10
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# train_images = train_images.reshape((60000, 32, 32, 3))
# test_images = test_images.reshape((10000, 32, 32, 3))
train_images, test_images = train_images / 255.0, test_images / 255.0

classes = list(range(10))

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) # alterado para 3 por serem imagens RGB
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(x = train_images, 
          y = train_labels,
          epochs=5, 
          validation_data=(test_images, test_labels),
          callbacks=[tensorboard_callback])


y_pred = np.argmax(model.predict(test_images), axis= -1)
y_true = test_labels.flatten()

con_mat = tf.math.confusion_matrix(labels = y_true, predictions = y_pred).numpy()

con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

con_mat_df = pd.DataFrame(con_mat_norm,
                          index = classes,
                          columns= classes)

figure = plt.figure(figsize=(8,8))
sns.heatmap(con_mat_df, annot=True, cmap= plt.cm.Blues, fmt= '.2f')
plt.tight_layout()
plt.xlabel("Rótulo Previsto")
plt.ylabel("Rótulo Verdadeiro")
plt.show()

# Cálculo das métricas de avaliação para um exemplo arbitrário (classificação binária)
'''
              Predito
            Positivo   Negativo
Verdadeiro
Positivo      VP = 40    FN = 10
Negativo      FP = 15    VN = 35
'''
VP = 40 # VP = 40 (Verdadeiros Positivos)
FN = 10 # FN = 10 (Falsos Negativos)
FP = 15 # FP = 15 (Falsos Positivos)
VN = 35 # VN = 35 (Verdadeiros Negativos)

sensibilidade = VP / (VP + FN)
especificidade = VN / (VN + FP)
acuracia = (VP + VN) / (VP + FN + FP + VN)
precisao = VP / (VP + FP)
f_score = 2 * (precisao * sensibilidade) / (precisao + sensibilidade)

print("\nMétricas com base na matriz de confusão arbitrária (classificação binária):")
print("Sensibilidade (Recall):", sensibilidade)
print("Especificidade:", especificidade)
print("Acurácia:", acuracia)
print("Precisão:", precisao)
print("F-score:", f_score)



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
