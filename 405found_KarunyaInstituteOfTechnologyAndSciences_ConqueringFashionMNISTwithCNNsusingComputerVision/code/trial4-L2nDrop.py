import tensorflow as tf
from tensorflow import keras
from keras import layers
import time

# Load the Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

# Normalize pixel values to a range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape the images to match the expected input shape of the model
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))

# Convert the labels to integers
train_labels = train_labels.astype(int)
test_labels = test_labels.astype(int)

# Define the model architecture with both L2 regularization and dropout regularization
model = keras.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1),
                  kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu',
                  kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu',
                 kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define a file to write the log
log_file = open('training_log_both.txt', 'w')

# Define a custom callback to log time and accuracy after each epoch
class LogCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        accuracy = logs['accuracy']
        log_file.write(f"Epoch {epoch+1} - Time: {current_time} - Accuracy: {accuracy:.4f}\n")
        log_file.flush()

# Train the model
start_time = time.time()
model.fit(train_images, train_labels, epochs=20, batch_size=128, callbacks=[LogCallback()])
end_time = time.time()
total_time = end_time - start_time

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
log_file.write(f"\nTest Loss: {test_loss:.4f}\n")
log_file.write(f"Test Accuracy: {test_accuracy:.4f}\n")
log_file.write(f"Total Training Time: {total_time:.2f} seconds\n")
log_file.close()

# Save the model
model.save('fashion_mnist_model_both.h5')
