import numpy as np
import random
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, merge, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, \
    LearningRateScheduler, TensorBoard
from keras.optimizers import Adam, SGD, RMSprop
import os
from keras.models import Model, load_model, Sequential
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from sklearn.metrics import accuracy_score as accuracy
import cv2
from keras import backend as K
from keras.regularizers import l2
from random import shuffle
import datetime
from tensorboard import program


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


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 ** (epoch / s)

    return exponential_decay_fn


def fit_model(model, path_to_model, train_pairs, train_label, batch_size, epochs, val_pairs,
              val_label,
              learning_rate, track_address=os.path.join(os.getcwd(), 'tensorboard')):
    """Fit the model"""

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', track_address])
    url = tb.launch()

    mcp = ModelCheckpoint(path_to_model, monitor='val_acc', verbose=1, save_best_only=True,
                          mode='max')
    logs = CSVLogger(os.path.join(path_to_model, 'logs/fit/log'))

    exponential_decay_fn = exponential_decay(lr0=learning_rate, s=20)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    lr_scheduler = LearningRateScheduler(exponential_decay_fn)
    model.compile(loss='binary_crossentropy', optimizer=Adam, metrics=['accuracy'])
    history = model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_label,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_label),
                        callbacks=[mcp, logs, lr_scheduler, tensorboard_callback])
    del model
    model = load_model(path_to_model)
    return model, history
