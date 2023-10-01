import tensorflow as tf

from tensorflow.keras import layers, models, utils

training_data, testing_data = utils.image_dataset_from_directory(
    "Data/",
    image_size=(600,600),
    seed=123456789,
    validation_split=0.2,
    subset="both"
)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(600, 600, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPool2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dense(7))

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y

training_data = training_data.map(preprocess)
testing_data = testing_data.map(preprocess)

history = model.fit(
    training_data, epochs=10, validation_data=testing_data
)
