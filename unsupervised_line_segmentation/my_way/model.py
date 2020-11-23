import numpy as np
import random
from keras.layers import Input, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras.optimizers import Adam
import os
from keras.models import Model, load_model
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import cv2
import re


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def genereate_batch(path_1, path_2, batch_size):
    """Generate batch of images from two files, where name corresponds to each other. Label is the last number in the name of the picture"""
    X = []
    labels = []
    dir_list_X1 = os.listdir(path_1)
    dir_list_X1.sort(key=natural_keys)
    dir_list_X2 = os.listdir(path_2)
    dir_list_X2.sort(key=natural_keys)
    batch_count = 0
    while True:
        for imgp in dir_list_X1:
            label = imgp[-5]  # eg. "SampleNb_1.png, that is why [-5] is "1"
            p1 = cv2.imread(os.path.join(path_1, imgp))
            p1 = cv2.cvtColor(p1, cv2.COLOR_BGR2GRAY)
            p1 = p1.reshape(p1.shape[0], p1.shape[1], 1)
            p2 = cv2.imread(os.path.join(path_2, imgp))
            p2 = cv2.cvtColor(p2, cv2.COLOR_BGR2GRAY)
            p2 = p2.reshape(p2.shape[0], p2.shape[1], 1)
            X += [[p1, p2]]
            labels += [label]
            batch_count += 1
            if batch_count > batch_size - 1:
                apairs = np.array(X, dtype=object)
                alabels = np.array(labels)
                yield [apairs[:, 0], apairs[:, 1]], alabels
                X.clear()
                labels.clear()
                batch_count = 0


def create_base_model(input_dim):
    """Create model first part based on input dimensions size"""
    inputs = Input(shape=input_dim)
    conv_1 = Conv2D(64, (5, 5), padding="same", activation='relu', name='conv_1')(inputs)
    conv_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
    conv_2 = Conv2D(128, (5, 5), padding="same", activation='relu', name='conv_2')(conv_1)
    conv_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
    conv_3 = Conv2D(256, (3, 3), padding="same", activation='relu', name='conv_3')(conv_2)
    conv_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
    conv_4 = Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_4')(conv_3)
    conv_5 = Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_5')(conv_4)
    conv_5 = MaxPooling2D(pool_size=(2, 2))(conv_5)

    dense_1 = Flatten()(conv_5)
    dense_1 = Dense(512, activation="relu")(dense_1)
    dense_1 = Dropout(0.5)(dense_1)
    dense_2 = Dense(512, activation="relu")(dense_1)
    dense_2 = Dropout(0.5)(dense_2)
    return Model(inputs, dense_2)


def built_model(input_shape):
    base_network = create_base_model(input_shape)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    fc6 = concatenate([processed_a, processed_b])
    fc7 = Dense(1024, activation='relu')(fc6)
    fc8 = Dense(1024, activation='relu')(fc7)
    fc9 = Dense(1, activation='sigmoid')(fc8)
    model = Model([input_a, input_b], fc9)
    model.summary()
    return model


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 ** (epoch / s)

    return exponential_decay_fn


def fit_model(model, path_to_model, generator_train, train_steps, generator_val, val_steps, epochs,
              learning_rate):
    """Fit the model"""

    mcp = ModelCheckpoint(os.path.join(path_to_model, 'bestmodel.h5py'), monitor='val_accuracy',
                          verbose=1,
                          save_best_only=True,
                          mode='max')
    logs = CSVLogger('learned_model/log')

    exponential_decay_fn = exponential_decay(lr0=learning_rate, s=20)

    lr_scheduler = LearningRateScheduler(exponential_decay_fn)
    adam = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    history = model.fit(generator_train,
                        steps_per_epoch=train_steps,
                        epochs=epochs,
                        validation_data=generator_val,
                        validation_steps=val_steps, shuffle=False,
                        callbacks=[logs, mcp, lr_scheduler])
    del model
    model = load_model(os.path.join(path_to_model, 'bestmodel.h5py'))
    return model, history
