# to load data in required manner
from formate_data import formate_data

#import basic neural net
import numpy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.models import Sequential
from keras.layers import Dense

#import CNN
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

#import for RNN/LSTM model
import pandas
import math
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error



# def create_dataset(dataset, look_back=1):
# 	dataX, dataY = [], []
# 	for i in range(len(dataset)-look_back-1):
# 		a = dataset[i:(i+look_back), 0]
# 		dataX.append(a)
# 		dataY.append(dataset[i + look_back, 0])
# 	return numpy.array(dataX), numpy.array(dataY)


numpy.random.seed(7)

matrix = formate_data()

size = 25
max_size = 31

# X_train = matrix[0:size+1, :, :]
X_train = matrix[0:size+1, :, :, :]
# X_test = matrix[size:max_size, :, :]
X_test = matrix[size:max_size, :, :, :]
# Y_train = numpy.delete(matrix, (0), axis=0)
# Y_train = numpy.sum(numpy.delete(X_train, (0), axis=0), axis=1)
# Y_train = numpy.sum(Y_train, axis=1)
# Y_train = numpy.sum(numpy.delete(X_train, (0), axis=0), axis=2)
# Y_train = numpy.sum(Y_train, axis=2)
Y_train = numpy.delete(X_train, (0), axis=0)


# X_train = matrix[0:size, :, :]
X_train = matrix[0:size, :, :, :]

# Y_test = numpy.sum(numpy.delete(X_test, (0), axis=0), axis=1)
# Y_test = numpy.sum(Y_test, axis=1)
# Y_test = numpy.sum(numpy.delete(X_test, (0), axis=0), axis=2)
# Y_test = numpy.sum(Y_test, axis=2)
Y_test = numpy.delete(X_test, (0), axis=0)

# X_test = matrix[size:max_size, :, :]
X_test = matrix[size:max_size-1, :, :, :]

# Y_train = numpy.reshape(Y_train, (-1, 1))
# Y_test = numpy.reshape(Y_test, (-1, 1))
# print "size of y = %s"%Y_train.shape

num_classes = Y_train.shape[1]


# Y_train = Y_train[24, :]
## a simple 3 layer neural net
# model = Sequential()
# model.add(Dense(12, input_dim=8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# vision_model = Sequential()
# vision_model.add(Conv2D(640, (3, 3), activation='relu', padding='same', input_shape=(2000, 2000, 1)))
# vision_model.add(Conv2D(640, (3, 3), activation='relu'))
# vision_model.add(MaxPooling2D((2, 2)))
# vision_model.add(Conv2D(1280, (3, 3), activation='relu', padding='same'))
# vision_model.add(Conv2D(1280, (3, 3), activation='relu'))
# vision_model.add(MaxPooling2D((2, 2)))
# vision_model.add(Flatten())

# # image_input = Input(shape=(3, 224, 224))
# image_input = X_train
# encoded_image = vision_model(image_input)

# # question_input = Input(shape=(100,), dtype='int32')
# question_input = Y_train
# embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=200)(question_input)
# encoded_question = LSTM(256)(embedded_question)

# merged = keras.layers.concatenate([encoded_question, encoded_image])

# output = Dense(1000, activation='softmax')(merged)

# vqa_model = Model(inputs=[image_input, question_input], outputs=output)

div_x = 32
div_y = 32

# implementation of CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(1, div_x, div_y)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
model.add(Flatten())
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.5))
# model.add(Dense(num_classes=400, activation='softmax'))
model.add(Dense(div_x*div_y, activation='softmax'))

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
model.add(Reshape((div_x, div_y)))
model.add(Reshape((1, div_x, div_y)))
model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same', input_shape=(1, div_x, div_y)))
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (30, 30), input_shape=(30, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))


# Compile model
epochs = 100
lrate = 0.1
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
# model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# Fit the model
# model.fit(X_train, Y_train, epochs=epochs, batch_size=25)
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=5)

# scores = model.evaluate(X_test, Y_test)
# print("\n%s: %.2f%%"%(model.metrics_names[1],scores[1]*100))

