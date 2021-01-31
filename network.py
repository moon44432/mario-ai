
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten
from keras.models import Sequential
from keras.regularizers import l2
import os

# params
img_width = 128
action_size = 5

DN_INPUT_SHAPE = (img_width, img_width, 4)
DN_OUTPUT_SIZE = action_size


def create_network():
    model = Sequential()

    model.add(Conv2D(16, (5, 5), activation='relu', padding='same', strides=2,
                     kernel_initializer='he_normal', kernel_regularizer=l2(0.0005), input_shape=DN_INPUT_SHAPE))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (5, 5), activation='relu', padding='same', strides=2,
                     kernel_initializer='he_normal', kernel_regularizer=l2(0.0005)))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (5, 5), activation='relu', padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=l2(0.0005)))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

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
