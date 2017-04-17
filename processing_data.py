# to load data in required manner
from formate_data import formate_data

#import basic neural net
import numpy
from keras.models import Sequential
from keras.layers import Dense

#import CNN
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

#import for RNN/LSTM model
import pandas
import math
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


numpy.random.seed(7)

X_train = matrix
y_train = numpy.delete(matrix, (0), axis=0)
## a simple 3 layer neural net
# model = Sequential()
# model.add(Dense(12, input_dim=8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

## implementation of CNN
model = Sequential()
model.add(Conv2D(32, (30, 30), input_shape=(30, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

## adding RNN Components

# # defining data set
# look_back = 1
# trainX, trainY = create_dataset(train, look_back)


## adding another reverse CNN
# model.add(Dense(num_classes, activation='softmax'))
# model.add(Dropout(0.5))
# model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Flatten())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (30, 30), input_shape=(30, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))


# Compile model
epochs = 10
lrate = 0.1
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)



