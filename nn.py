import numpy as np 
from tkinter import *
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout

### LOAD DATASET ###
# dataset = matrix of 5 columns with input variables and 1 column with output variables
# x_train = [:, 0:6]
# y_train = [:,6]

### CREATE MODEL ###
model = Sequential()
model.add(Dense(16, input_dim=5, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='softmax'))


model.compile(loss='mean_squared_error', optimizer='adam', metrics='accuracy')

### TRAIN THE MODEL ###
#model.fit(x_train, y_train, epochs=100)

#accuracy = model.evaluate(x_train, y_train)
#print('Accuracy: %.2F' % (accuracy*100))

### MAKE PREDICTIONS ###: 
# p > 0.995 (high risk of infection -> isolate without ), 0.5 < p < 0.995 
# (medium risk of infection -> highest temp individuals get tested) 
# and p < 0.5 (low risk of infection)

#predictions = model.predict(x_train)
#if (p > 0.995): 
#   isolate
#if (0.5 < p < 0.995): 
#   hi_temp_individuals.test()
#if (p < 0.5): 
#   ???
