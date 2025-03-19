import joblib
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=500,  # Number of trees in the forest
            max_depth=None,    # Maximum depth of the trees
            max_features='sqrt',
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,   # For reproducibility
            n_jobs=-1,
            verbose=2
        )
    
    def loadData(self):
        # NumPy arrays
        (trainImages, trainLabels), (testImages, testLabels) = tf.keras.datasets.mnist.load_data()
        trainImagesReshaped = trainImages.reshape((trainImages.shape[0], -1))
        testImagesReshaped = testImages.reshape((testImages.shape[0], -1))

        # Normalize the images to the range [0, 1]
        trainImagesReshaped, testImagesReshaped = trainImagesReshaped / 255, testImagesReshaped / 255

        return (trainImagesReshaped, trainLabels), (testImagesReshaped, testLabels)

    def train(self, trainImages, trainLabels):
        # Train the model, use 20% of training data for validation
        return self.model.fit(trainImages, trainLabels)

    def evaluate(self, testImages, testLabels):
        # Sensitivity (recall - czułość), true_pos / (true_pos + false_neg)
        predictions = self.model.predict(testImages)
        print("\nClassification Report:")
        print(classification_report(testLabels, predictions))
        # Confusion matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(testLabels, predictions))

        return predictions
    
    def saveModel(self, filepath):
        joblib.dump(self.model, "keras_models/" + filepath)
        print("Model saved")

model = RandomForestModel()
(train_images, train_labels), (test_images, test_labels) = model.loadData()
history = model.train(train_images, train_labels)
predicted_classes = model.evaluate(test_images, test_labels)
model.saveModel("rf_model.joblib")


