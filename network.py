
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import os

# params
img_width = 128
action_size = 5

DN_INPUT_SHAPE = (img_width, img_width, 4)
DN_OUTPUT_SIZE = action_size


def set_network():
    if os.path.exists('./model/model.h5'):
        return

    model = Sequential()

    model.add(Conv2D(16, (5, 5), activation='relu', padding='same', strides=2,
                     kernel_initializer='he_normal', kernel_regularizer=l2(0.0005), input_shape=DN_INPUT_SHAPE))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (5, 5), activation='relu', padding='same', strides=2,
                     kernel_initializer='he_normal', kernel_regularizer=l2(0.0005)))
    model.add(Conv2D(32, (5, 5), activation='relu', padding='same', strides=2,
                     kernel_initializer='he_normal', kernel_regularizer=l2(0.0005)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(DN_OUTPUT_SIZE, activation='linear'))

    print(model.summary())

    os.makedirs('./model/', exist_ok=True)
    model.save('./model/model.h5')

    K.clear_session()
    del model

if __name__ == '__main__':
    set_network()
