import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the STL-10 dataset
stl10_dataset, dataset_info = tfds.load("stl10", with_info=True, as_supervised=True)
train_dataset = stl10_dataset['train']
test_dataset = stl10_dataset['test']

AUTOTUNE = tf.data.AUTOTUNE

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize
    print(image.shape)
    label = tf.cast(label, tf.int32)           # Cast to int32 for one-hot encoding
    label = tf.one_hot(label, depth=10)        # One-hot encode
    return image, label

train_dataset = train_dataset.map(preprocess, num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.map(preprocess, num_parallel_calls=AUTOTUNE)

# Add data augmentation
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_crop(image, size=[96, 96, 3])
    return image, label

train_dataset = train_dataset.map(augment, num_parallel_calls=AUTOTUNE)

# Prefetch to improve pipeline performance
train_dataset = train_dataset.shuffle(1000).batch(32).prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(buffer_size=AUTOTUNE)


from tensorflow.keras.layers import SeparableConv2D

# Define the optimized CNN model
model = Sequential([
    # Use SeparableConv2D for efficiency
    SeparableConv2D(64, (3, 3), activation='relu', input_shape=(96, 96, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    SeparableConv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(64, activation='relu'),  # Reduce dense layer size
    tf.keras.layers.Dropout(0.3),  # Reduce dropout rate for less regularization
    Dense(10, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# Prepare training loop with custom output
train_acc = []
val_acc = []
train_loss = []
val_loss = []
epochs = 50  # Adjust epochs as needed

for epoch in range(epochs):
    start_time = time.time()  # Record start time

    # Train the model for one epoch
    history = model.fit(
        train_dataset,
        epochs=1,
        validation_data=test_dataset,
        verbose=1
    )

    # Calculate the time taken for this epoch
    end_time = time.time()
    epoch_time = end_time - start_time

    # Get the accuracy and loss for both train and validation
    train_accuracy = history.history['accuracy'][0]
    val_accuracy = history.history['val_accuracy'][0]
    train_loss.append(history.history['loss'][0])
    val_loss.append(history.history['val_loss'][0])

    # Print the output in the desired format
    print(f"Epoch {epoch + 1}/{epochs} - Time: {epoch_time:.2f}s - Train Acc: {train_accuracy:.4f} - Val Acc: {val_accuracy:.4f}")

    # Save accuracy data for later
    train_acc.append(train_accuracy)
    val_acc.append(val_accuracy)

# Save accuracy and loss data to an Excel file
data = {
    "Epoch": list(range(1, epochs + 1)),
    "Training Accuracy": train_acc,
    "Validation Accuracy": val_acc,
    "Training Loss": train_loss,
    "Validation Loss": val_loss
}
df = pd.DataFrame(data)
df.to_excel("stl10_accuracy_data.xlsx", index=False)
print("Accuracy and loss data saved to 'stl10_accuracy_data.xlsx'")

# Plot the training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_acc, label='Training Accuracy', marker='o')
plt.plot(range(1, epochs + 1), val_acc, label='Validation Accuracy', marker='o')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the model on the test set
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Make predictions
predictions = model.predict(test_dataset)

# Visualize predictions with ground truth
plt.figure(figsize=(12, 12))
for i, (image, label) in enumerate(test_dataset.unbatch().take(25)):
    plt.subplot(5, 5, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    pred = np.argmax(predictions[i])
    groundtruth = tf.argmax(label).numpy()
    col = 'g' if pred == groundtruth else 'r'
    plt.xlabel(f'Pred: {pred} | GT: {groundtruth}', color=col)
    plt.imshow(image.numpy())
plt.tight_layout()
plt.show()
