import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow_datasets as tfds
import numpy as np

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

x_train_reshape=np.reshape(x_train,(5000,27648))
x_test_reshape=np.reshape(x_test,(8000,27648))

#normalization:

x_mean=np.mean(x_train_reshape)
x_std=np.std(x_train_reshape)

epsilon=1e-10

x_train_norm=(x_train_reshape-x_mean)/(x_std+epsilon)
x_test_norm=(x_test_reshape-x_mean)/(x_std+epsilon)

#print(set(x_train_norm[0]))
model= Sequential([Dense(256, activation='relu', input_shape=[27648,]), Dense(256, activation='relu'), Dense(10, activation='softmax')])

#compiling model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


#training the model
model.fit(x_train_norm, y_train_encoded, epochs=10, batch_size=32, validation_data=(x_test_norm, y_test_encoded))

#evaluating model:

loss, accuracy=model.evaluate(x_test_norm, y_test_encoded)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


#predictions
predictions=model.predict(x_test_norm)
#print(predictions.shape)
