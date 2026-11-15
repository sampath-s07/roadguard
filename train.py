from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os


def run_training(model, xtr, ytr, xv, yv, name="model", epochs=35, bs=64):
    os.makedirs("saved_models", exist_ok=True)

    ck = ModelCheckpoint(
        f"saved_models/{name}.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    es = EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    h = model.fit(
        xtr, ytr,
        batch_size=bs,
        validation_data=(xv, yv),
        epochs=epochs,
        callbacks=[ck, es],
        verbose=1
    )
    return h
