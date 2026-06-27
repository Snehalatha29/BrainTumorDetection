import tensorflow as tf

# Load Training Dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/Training",
    image_size=(224, 224),
    batch_size=32,
    shuffle=True
)

# Load Testing Dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/Testing",
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

print("Training Dataset Loaded Successfully!")
print("Testing Dataset Loaded Successfully!")

model = tf.keras.Sequential([

    # First Convolution Layer
    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    # Second Convolution Layer
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    # Third Convolution Layer
    tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3,3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(
        4,
        activation="softmax"
    )

])

print("CNN Model Created Successfully!")
# Compile the Model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Model Compiled Successfully!")
# Compile Model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Model Compiled Successfully!")
# Train Model
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=10
)

print("Model Training Completed!")
# Save Model
model.save("brain_tumor_model.keras")

print("Model Saved Successfully!")