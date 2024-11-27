import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

#Load the STL-10 dataset

stl10_dataset, dataset_info = tfds.load("stl10", with_info=True, as_supervised=True)

#Split the dataset into train and test
train_dataset = stl10_dataset['train']
test_dataset = stl10_dataset['test']

#Convert datasets to arrays
def dataset_to_numpy(dataset):
    images = []
    labels = []
    for image, label in tfds.as_numpy(dataset):
        images.append(image)
        labels.append(label)
    return tf.convert_to_tensor(images), tf.convert_to_tensor(labels)

x_train, y_train = dataset_to_numpy(train_dataset)
x_test, y_test = dataset_to_numpy(test_dataset)

#one hot encoding
print("y_train:", y_train)
print("y_test:", y_test)

y_train_encoded = to_categorical(y_train, num_classes=10)
y_test_encoded = to_categorical(y_test, num_classes=10)

# Normalize: Cast x_train and x_test to float32 before dividing by 255.0
x_train_norm = tf.cast(x_train, tf.float32) / 255.0
x_test_norm = tf.cast(x_test, tf.float32) / 255.0

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import BatchNormalization
datagen = ImageDataGenerator(
    rotation_range=10,  
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    shear_range=0.1,  
    zoom_range=0.1,  
    horizontal_flip=True,
    fill_mode='nearest'
)



#print(set(x_train_norm[0]))

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])



#compiling model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


# Fit the model with augmented data
model.fit(datagen.flow(x_train_norm, y_train_encoded, batch_size=32), epochs=40, validation_data=(x_test_norm, y_test_encoded))

#evaluating model:

loss, accuracy=model.evaluate(x_test_norm, y_test_encoded)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


#predictions
predictions=model.predict(x_test_norm)
#print(predictions.shape)
