# Conquering Fashion MNIST dataset with CNNs
## OBJECTIVE:
The objective of this project is to develop a Convolutional Neural Network (CNN) model to achieve high accuracy in classifying the Fashion MNIST dataset. The goal is to surpass a test accuracy of 92% using TensorFlow and leverage Intel optimizations for improved performance.
## EXISTING METHODS :
1.	Fashion MNIST Dataset: The Fashion MNIST dataset is a collection of 70,000 grayscale images categorized into 10 different clothing classes. It is widely used as a benchmark for image classification tasks due to its similarity to the original MNIST dataset.
2.	Convolutional Neural Networks (CNNs): CNNs have proven to be highly effective for image classification tasks. They consist of multiple convolutional layers for feature extraction and pooling layers for down sampling, followed by fully connected layers for classification.
## The Project Model consists of the following LAYERS:
1.	Convolution: Convolution involves sliding the filter over the input image, element-wise multiplying the values of the filter with the corresponding values of the image patch covered by the filter, and summing up the results.
2.	ReLU: ReLU short for Rectified Linear Unit, is an activation function commonly used in neural networks, including convolutional neural networks (CNNs). It introduces non-linearity to the network, allowing it to learn complex patterns and make the model more expressive.
3.	Pooling:  Pooling also known as subsampling or down sampling, is a common operation in convolutional neural networks (CNNs) used to reduce the spatial dimensions of feature maps. It aims to extract the most important and representative information while decreasing the computational requirements and introducing a degree of translation invariance.
4.	Fully Connected layer:  A Fully Connected layer also known as a Dense layer, is a fundamental component in artificial neural networks, including convolutional neural networks (CNNs). It is responsible for connecting every neuron from the previous layer to every neuron in the current layer, creating a fully connected network structure.
5.	Dropout regularization: Dropout regularization is a technique used in neural networks, including convolutional neural networks (CNNs), to prevent overfitting and improve generalization performance. It involves randomly disabling or "dropping out" a proportion of neurons in a layer during training, forcing the network to learn redundant representations and reducing co-adaptation between neurons.

## RESULT:
Final (Dropout.>epochs)	0.9185	0.9196

The model achieved a test accuracy of 92% after training on the Fashion MNIST dataset consisting of 60,000 images and testing on 10,000 images. This high accuracy demonstrates the effectiveness of the implemented CNN architecture in accurately classifying clothing items.

The training and testing process revealed the following insights:

-	Training time: The model took approximately X hours to train on an Intel DevCloud platform.
-	Epoch-wise accuracy: The accuracy of the model improved steadily with each epoch, as shown in the accuracy log file.
-	Overfitting: By incorporating dropout regularization, the model effectively reduced overfitting and achieved higher generalization performance.
