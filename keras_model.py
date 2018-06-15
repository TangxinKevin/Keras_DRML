from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D
from keras.layers import Flatten, Dropout, Dense

def DRML(input_shape, num_outputs):
    img_input = Input(shape=input_shape)
    x = Conv2D(32, (11, 11), activation='relu', name='conv1')(img_input)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv2')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool3')(x)
    x = Conv2D(16, (8, 8), activation='relu', name='conv4')(x)
    x = Conv2D(16, (8, 8), activation='relu', name='conv5')(x)
    x = Conv2D(16, (6, 6), strides=(2, 2), activation='relu', name='conv6')(x)
    x = Conv2D(16, (5, 5), activation='relu', name='conv7')(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(4096, activation='relu', name='fc8')(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu', name='fc9')(x)
    x = Dropout(0.5)(x)
    out = Dense(num_outputs, activation='sigmoid', name='output')(x)

    model = Model(inputs=img_input, outputs=out)
    return model

