from tensorflow.keras import Model
from tensorflow.keras.layers import Input, InputLayer, Conv2D, Dropout, MaxPooling2D, concatenate, UpSampling2D

def ShapesModel(classes=4, shape=(128, 128)):
    if classes == 1:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    inputs = Input(shape=(128, 128, 1))

    # Encoder
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    # Decoder
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = UpSampling2D((2, 2))(conv3)

    up1 = concatenate([conv3, conv2], axis=-1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = UpSampling2D((2, 2))(conv4)

    up2 = concatenate([conv4, conv1], axis=-1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

    outputs = Conv2D(classes, (1, 1) , padding='same')(conv5)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model