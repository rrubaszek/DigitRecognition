import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report #type: ignore
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  #type: ignore

class ConvolutionalModel:
    def __init__(self):
        # Model with sequential layers, each with exactly one input and output
        self.model = Sequential() 

        # First convolutional layer to capture low-level patterns in numbers
        # (3, 3) kernel size to capture enough details, but without the risk of overfitting
        # (1, 1) strides (default), dataset is balanced, don't need downsampling
        self.model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation ='relu')) 

        # (2, 2) pool size reduces computational costs and preserves details, from 28x28 to 14x14
        # (2, 2) strides moves by 2 pixels and ensures no overlapping
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) 

        # Another convolutional layer extracts more complex patterns in numbers
        # All settings stays the same, we only use more filters (64)
        self.model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation ='relu')) 

        # Second pooling layer, reduces to 7x7, prevents overfitting by further reducing details
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) 

        # Converts 7x7x64 map to 1D vector
        self.model.add(Flatten()) 

        # Fully connected layer of neurons y = wx + b, combines features of previous layers and makes the decision
        self.model.add(Dense(1000, activation='relu')) 

        # Each class (0-9) has its own neuron, softmax -> probability distribution
        self.model.add(Dense(10, activation='softmax')) 
        
        # Best loss function for multi-class classification
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics =['accuracy']) 

        print("Model constructed")
    
    def loadData(self):
        # NumPy arrays
        (trainImages, trainLabels), (testImages, testLabels) = tf.keras.datasets.mnist.load_data()
        trainImagesReshaped = trainImages.reshape((trainImages.shape[0], 28, 28, 1))
        testImagesReshaped = testImages.reshape((testImages.shape[0], 28, 28, 1))

        # Normalize the images to the range [0, 1]
        trainImagesReshaped, testImagesReshaped = trainImagesReshaped / 255, testImagesReshaped / 255

        return (trainImagesReshaped, trainLabels), (testImagesReshaped, testLabels)

    def train(self, trainImages, trainLabels, epochs=5, batch_size=64, validation_split=0.2):
        # Train the model, use 20% of training data for validation
        return self.model.fit(trainImages, trainLabels, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def evaluate(self, testImages, testLabels):
        # Accuracy (współczynnik prawidłowej rozpoznawalności)
        testLoss, testAccuracy = self.model.evaluate(testImages, testLabels, verbose=2)
        print(f"Test accuracy: {testAccuracy * 100:.2f}%")

        # Sensitivity (recall - czułość), true_pos / (true_pos + false_neg)
        predictions = self.model.predict(testImages)
        predictedClasses = predictions.argmax(axis=1)
        print(classification_report(testLabels, predictedClasses))

        return testLoss, testAccuracy, predictions, predictedClasses

    def saveModel(self, filepath):
        self.model.save("keras_models/" + filepath)
        print("Model saved")

    def predict(self, image):
        predictions = self.model.predict(image)
        predicted_digits = np.argmax(predictions, axis=1)
        return predicted_digits

model = ConvolutionalModel()
(train_images, train_labels), (test_images, test_labels) = model.loadData()
history = model.train(train_images, train_labels)
loss, accuracy, predictions, predicted_classes = model.evaluate(test_images, test_labels)
model.saveModel("conv_model.keras")