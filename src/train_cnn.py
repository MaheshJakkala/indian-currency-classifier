from utils import *

def cnn_model_train():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(7, activation="softmax")
    ])
    model.summary()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    train_datagen = ImageDataGenerator(rescale=1/255)
    validation_datagen = ImageDataGenerator(rescale=1/255)

    # Flow training images in batches of 128 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(main_train_dir,
                                                    batch_size=64,
                                                    target_size=(150,150),
                                                    class_mode="categorical")

    validation_generator = validation_datagen.flow_from_directory(main_test_dir,
                                                                batch_size=16,
                                                                target_size=(150,150),
                                                                class_mode="categorical")
    history = model.fit(train_generator,
                        epochs=100,
                        steps_per_epoch=len(train_generator),
                        verbose=1,
                        validation_data=validation_generator,
                        validation_steps=len(validation_generator))
    
    model.save("models/cnn_currency.h5")

def plot_training_history(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(len(acc))

    plt.plot(epochs, acc, "r", label="Training Accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation Accuracy")

    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, "r", label="Training Loss")
    plt.plot(epochs, val_loss, "b", label="Validation Loss")

    plt.legend()
    plt.show()