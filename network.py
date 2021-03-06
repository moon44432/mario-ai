import os

from keras.layers import Conv2D, Dense, Flatten, Dropout, BatchNormalization, Activation
from keras.models import Sequential
from keras.regularizers import l2

from env import action_size
from hparams import state_deque_size
from process_image import img_width

DN_INPUT_SHAPE = (img_width, img_width, state_deque_size)
DN_OUTPUT_SIZE = action_size


def create_network():
    model = Sequential()

    model.add(Conv2D(32, (8, 8), padding='same', strides=4,
                     kernel_initializer='he_normal', kernel_regularizer=l2(0.0005), input_shape=DN_INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(64, (4, 4), padding='same', strides=2,
                     kernel_initializer='he_normal', kernel_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), padding='same', strides=1,
                     kernel_initializer='he_normal', kernel_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(DN_OUTPUT_SIZE, activation='linear'))

    print(model.summary())

    return model


def set_network():
    if os.path.exists('./model/model.h5'):
        return

    model = create_network()

    os.makedirs('./model/', exist_ok=True)
    model.save('./model/model.h5')

    del model


if __name__ == '__main__':
    set_network()
