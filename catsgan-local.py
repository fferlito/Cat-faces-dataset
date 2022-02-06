# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 19:32:11 2022

@author: aaaro

"""

import os

os.chdir('Documents\\GitHub\\Cat-faces-dataset\\')

import tarfile

def unzip_files(file_name, destination_folder):
    file = tarfile.open(f'''{file_name}.tar.gz''')
    file.extractall(f'''./{destination_folder}''')
    file.close()

unzip_files(file_name = 'dataset-part1',
             destination_folder = './cat-faces')

unzip_files(file_name = 'dataset-part2',
             destination_folder = './cat-faces')

unzip_files(file_name = 'dataset-part3',
             destination_folder = './cat-faces')



import numpy as np
import PIL
import PIL.Image
import matplotlib.pyplot as plt

file_names = ['cat-faces/dataset-part1/' + name for name in list(os.listdir('cat-faces/dataset-part1/'))][1000:]
cats = [np.array(PIL.Image.open(file_name)) for file_name in file_names]
labels = [1 for _ in range(len(cats))]

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) >= 1

cats_tf = tf.data.Dataset.from_tensor_slices((cats, labels))


train_ds = tf.keras.utils.image_dataset_from_directory('cat-faces/dataset-part1/',
                                                        seed=123,
                                                        image_size=(64, 64),
                                                        color_mode = 'rgb',
                                                        label_mode = 'categorical',
                                                        batch_size=32)


training_cats = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input) \
                    .flow_from_directory(directory = 'cat-faces/dataset-part1/train',
                                         target_size = (64, 64),
                                         classes = ['cat'],
                                         batch_size = 10)
                    
training_cats = tf.reshape(training_cats, (-1, 64, 64, 3))
                
def plot_cats(images_arr):
    
    fig, axes = plt.subplots(1, 10, figsize = (10, 10))
    
    axes = axes.flatten()
    
    for img, ax in zip(images_arr, axes):
        
        ax.imshow(img)
        
        ax.axis('off')
        
    plt.tight_layout()
    
    plt.show()
    
    return
    

def get_sample_cat_batch(training_cats):
    
    return(next(training_cats))

batch_of_cats, labels = get_sample_cat_batch(training_cats)

batch_of_cats = tf.reshape(batch_of_cats, (-1, 64, 64, 3))

training_cats = tf.expand_dims(training_cats, axis=-1)

plot_cats(batch_of_cats)

#   https://www.tensorflow.org/tutorials/generative/dcgan#the_generator

def make_generator(nc = 3, nz = 5):
    
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(64,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((7, 7, 256)))

    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    
    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    
    model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    
    return(model)

generator = make_generator()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')


print(tf.random.normal(shape = (5,)))    

def make_discriminator():
    
    model = tf.keras.Sequential()
    
  #  model.add(tf.keras.layers.Reshape((-1, 64, 64), input_shape = (64, 64, 3)))
    
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[64, 64, 3]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides = (2,2), padding = 'same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
    
    return(model)
    

discriminator = make_discriminator()

decision = discriminator(batch_of_cats[0])

print(decision)

        

