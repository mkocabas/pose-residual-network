import keras

from keras.models import Model
from keras.layers import Dense, Input, Flatten, Reshape, Dropout, Activation

def PRN(height, width, node_count):
    input = Input(shape=(height, width, 17))
    y = Flatten()(input)
    x = Dense(node_count, activation='relu')(y)
    x = Dropout(0.5)(x)
    x = Dense(width * height * 17, activation='relu')(x)
    x = keras.layers.Add()([x, y])
    x = keras.layers.Activation('softmax')(x)
    x = Reshape((height, width, 17))(x)
    model = Model(inputs=input, outputs=x)
    print model.summary()
    return model


def PRN_Seperate(height, width, node_count):
    input = Input(shape=(height, width, 17))
    y = Flatten()(input)
    x = Dense(node_count, activation='relu')(y)
    x = Dropout(0.5)(x)
    x = Dense(width * height * 17, activation='relu')(x)
    x = keras.layers.Add()([x, y])
    out = []
    start = 0
    end = width * height

    for i in range(17):
        o = keras.layers.Lambda(lambda x: x[:, start:end])(x)
        o = Activation('softmax')(o)
        out.append(o)
        start = end
        end = start + width * height

    x = keras.layers.Concatenate()(out)
    x = Reshape((height, width, 17))(x)
    model = Model(inputs=input, outputs=x)
    print model.summary()
    return model