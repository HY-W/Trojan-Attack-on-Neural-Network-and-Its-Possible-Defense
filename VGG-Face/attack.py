import numpy as np
import tensorflow as tf
import keras
import PIL
import pandas as pd
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras_vggface.vggface import VGGFace
from keras.preprocessing.image import ImageDataGenerator

vgg_model = VGGFace(include_top=True, input_shape=(224, 224, 3))

for layer in vgg_model.layers:
    layer.trainable = False
last_layer = vgg_model.get_layer('fc7/relu').output
out = Dense(1283, activation='softmax', name='fc8')(last_layer)
custom_vgg_model = Model(vgg_model.input, out)

adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999)
custom_vgg_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
custom_vgg_model.load_weights('model_weight.h5')

traindir = r"..\data\VGG_DB\trainSet"

input_shape = ()
batch_size = 64
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

backdoor_path = 'data\VGG_DB'
blindset = backdoor_path + '\BackDoor'
blind_datagen = ImageDataGenerator(
    rescale = 1. / 255)

blind_generator = blind_datagen.flow_from_directory(
    directory=blindset,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle = False
)


filenames = blind_generator.filenames
nb_samples = len(filenames)

prediction = custom_vgg_model.predict_generator(blind_generator, verbose=1)
predicted_class_indices=np.argmax(prediction, axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
new_filenames=[]
for f in filenames:
  f=f.replace('images/','')
  new_filenames.append(f)

results=pd.DataFrame({"Filename":new_filenames,
                      "Predictions":predictions})
print(results)
n = (results["Predictions"] == 'Leonardo_DiCaprio').value_counts().tolist()[0]


success_rate = (n/21)*100
print('The attack success rate is ' + str(success_rate) + "%")
