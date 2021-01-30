
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, MaxPool2D, Flatten, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import os

# params
dn_filters = 32
dn_residual_num = 4
img_width = 128
action_size = 5

DN_INPUT_SHAPE = (img_width, img_width, 4)
DN_OUTPUT_SIZE = action_size


def set_network():
    if os.path.exists('./model/model.h5'):
        return

    input = Input(shape=DN_INPUT_SHAPE)

    x = Conv2D(32, (3, 3), activation='relu', padding='same'
               , kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))(input)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', strides=2
               , kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same'
               , kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', strides=2
               , kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same'
               , kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(256, kernel_regularizer=l2(0.0005), activation='relu')(x)
    x = Dropout(0.5)(x)

    p = Dense(DN_OUTPUT_SIZE, kernel_regularizer=l2(0.0005),
              activation='linear')(x)

    model = Model(inputs=input, outputs=p)

    print(model.summary())

    os.makedirs('./model/', exist_ok=True)
    model.save('./model/model.h5')

    K.clear_session()
    del model

if __name__ == '__main__':
    set_network()
