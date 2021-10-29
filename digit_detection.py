from keras.datasets import mnist
from keras import Sequential
from keras.layers import Dense, Input
import numpy as np
from keras.utils.all_utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Aus 28x28 Matrix wird 784x1 Vektor gemacht

# Schwarz-Weiß Pixel haben Werte von 0 bis 255 
# Normalisieren indem man alle Werte durch 255 teilt - Werte liegen zw. 0 und 1

# Wir haben es also mit 1D Werten gemacht, ich hab aber auch noch ein Model mit
# 2D Input gebaut - digit_detection_2d.py

# Danach einfach normales neuronales Netzwerk, ohne Verwendung von CNN Elementen

x_train = x_train.reshape([60000, 28*28])
x_test = x_test.reshape([10000, 28*28])
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# One Hot Encoding - die Werte in y_train und y_test liegen zwischen 0 und 9
# Aus jedem Wert y aus diesen Listen wird ein Array der Länge 10 (weil wir ja 10 Werte haben) gemacht
# Überall 0, außer an der Stelle y, da ist Wert 1

# Am Ende erhält man vom neuronalen Netzwerk die Wahrscheinlichkeiten für jede Katgorie
# Wahrscheinlichster Wert wird gewählt

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Model wird erstellt mit verschiedenen Layers

model = Sequential()
# Input Layer
model.add(Input(shape=(784,)))
# Hidden Layers
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
# Output Layer
model.add(Dense(10, activation='softmax'))

# Verschiedene Optimizer findest du im Link in der README.md Datei
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Model wird auf Datensatz trainiert
model.fit(x_train, y_train, epochs=5)

# Model wird auf Testdatensatz bewertet
result = model.evaluate(x_test, y_test)

print('Accuracy: ' + result[1])
