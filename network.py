
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import os

# params
dn_filters = 32
dn_residual_num = 4
img_width = 128
action_size = 10

DN_INPUT_SHAPE = (img_width, img_width, 4)
DN_OUTPUT_SIZE = action_size


def conv(filters):
    return Conv2D(filters, 3, padding='same', use_bias=False,
                  kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))


def residual_block():
    def f(x):
        sc = x
        x = conv(dn_filters)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = conv(dn_filters)(x)
        x = BatchNormalization()(x)
        x = Add()([x, sc])
        x = Activation('relu')(x)
        return x

    return f


def set_network():
    if os.path.exists('./model/model.h5'):
        return

    input = Input(shape=DN_INPUT_SHAPE)

    x = conv(dn_filters)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for i in range(dn_residual_num):
        x = residual_block()(x)

    x = GlobalAveragePooling2D()(x)

    p = Dense(DN_OUTPUT_SIZE, kernel_regularizer=l2(0.0005),
              activation='softmax', name='pi')(x)

    model = Model(inputs=input, outputs=p)

    os.makedirs('./model/', exist_ok=True)
    model.save('./model/model.h5')

    K.clear_session()
    del model

if __name__ == '__main__':
    set_network()
