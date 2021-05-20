#%%

import tensorflow as tf
import numpy as np
import os
import keras
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam
from keras_vggface import VGGFace
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping



device_name = tf.test.gpu_device_name()

vgg_face = VGGFace(include_top=True, input_shape=(224,224,3))

for layer in vgg_face.layers:
    layer.trainable = False

last_layer = vgg_face.get_layer('fc7/relu').output

out = Dense(1283, activation='softmax', name='fc8')(last_layer)
custom_vgg_face = Model(vgg_face.input, out)

adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999)
custom_vgg_face.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])




traindir = r"..\data\VGG_DB\trainSet"
valdir = r"..\data\VGG_DB\validSet"
testdir = r"..\data\VGG_DB\TestSet"

batch_size = 64
input_shape = ()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.25,
    zoom_range=0.25,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    traindir,
    target_size=(224,224),
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle = True
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.25,
    zoom_range=0.25,
    horizontal_flip=True
)

val_generator = train_datagen.flow_from_directory(
    valdir,
    target_size=(224,224),
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle = True
)


num_samples = train_generator.n
num_classes = train_generator.num_classes


print('Loaded %d training samples from %d classes.' %(num_samples,num_classes))
print('Loaded %d test samples from %d classes.' %(val_generator.n,val_generator.num_classes))


callbacks =  [EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
              ModelCheckpoint(filepath='model_weight.h5', monitor='val_accuracy', save_best_only=True, verbose=1)]
history = custom_vgg_face.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    callbacks = callbacks,
    validation_data = val_generator,
    validation_steps = val_generator.samples // batch_size,
    epochs = 50,
    verbose = 1)

custom_vgg_face.load_weights('model_weigth.h5')

loss, acc = custom_vgg_face.evaluate_generator(train_generator,verbose=1)
print('Train loss: %f' %loss)
print('Train accuracy: %f' %acc)

testdir = '..\VGG_DB\TestSet'
test_datagen = ImageDataGenerator(
    rescale=1./255,
    )

test_generator = test_datagen.flow_from_directory(
    testdir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle = False
    )

loss, acc = custom_vgg_face.evaluate_generator(test_generator,verbose=1)
print('Test loss: %f' %loss)
print('Test accuracy: %f' %acc)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()