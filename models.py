from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense,
    Flatten, Dropout, GlobalAveragePooling2D, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2


def build_cnn():
    m = Sequential([
        Input(shape=(32, 32, 3)),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.2),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.3),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.4),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(43, activation='softmax')
    ])
    m.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return m


def build_transfer():
    base = MobileNetV2(
        input_shape=(32, 32, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False
    inp = Input(shape=(32, 32, 3))
    x = base(inp, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    out = Dense(43, activation="softmax")(x)
    model = Model(inp, out)
    model.compile(
        optimizer=Adam(0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, base
