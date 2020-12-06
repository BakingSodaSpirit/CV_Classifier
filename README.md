# CV_Classifier

Background:<br />
Object classification is the critical task in computer vision applications. It is the task of classifying objects from different object categories. Deep learning approaches perform an astoundingly better accuracy in computer vision than traditional methods. Convolutional neural networks (CNNs), also known on ConvNets, are the subject of the programming assignment.

Training Data<br />
The datasets used is MNIST. The sets are 10-class systems supported with the Keras neural network package.<br />

MNIST<br />
The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning.<br />
The MNIST dataset consists of 60000 28x28 grayscale (1 channel) images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.<br />


Base Architecture<br />
Layer__Description<br />
Conv2D__32 3x3 kernels with ReLU activation<br />
MaxPooling2D__2x2 poolsize<br />
Conv2D__64 3x3 kernels with ReLU activation<br />
MaxPooling2D__2x2 poolsize<br />
Flatten__Interface between Conv2D and Fully Connected<br />
Dense__100, ReLU activation<br />
Dense__10-output layer, SoftMax activation<br />

SmallVGG is based on the VGG16 convolutional neural network “Very Deep Convolutional Networks for Large-Scale Image Recognition”. In 2016, the VGG16 model achieved 92.7% top-5 test accuracy in ImageNet (14 million images belonging to 1000 classes).<br />
https://neurohive.io/en/popular-networks/vgg16/<br />

Layer__Description<br />
Conv2D__32 3x3 kernels with ReLU activation<br />
MaxPooling2D__3x3 poolsize<br />
Conv2D__64 3x3 kernels with ReLU activation<br />
MaxPooling2D__3x3 poolsize<br />
Conv2D__64 3x3 kernels with ReLU activation<br />
MaxPooling2D__3x3 poolsize<br />
Conv2D__128 3x3 kernels with ReLU activation<br />
MaxPooling2D__3x3 poolsize<br />
Conv2D__128 3x3 kernels with ReLU activation<br />
MaxPooling2D__3x3 poolsize<br />
Flatten__Interface between Conv2D and Fully Connected<br />
Dense__256, ReLU activation<br />
Dense__10-output layer, SoftMax activation<br />

Evaluation<br />
confusion matrix:<br />
A confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known. The confusion matrix itself is relatively simple to understand, but the related terminology can be confusing.<br />
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html<br />

top-k prediction accuracy:<br />
https://sklearn-theano.github.io/auto_examples/plot_classification.html<br />
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html<br />

Sensitivity and Specificity:<br />
https://en.wikipedia.org/wiki/Sensitivity_and_specificity<br />

Precision and Recall:<br />
https://en.wikipedia.org/wiki/Precision_and_recall<br />
https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html<br />
