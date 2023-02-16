# An Overview of Machine Learning Techniques

Basics of machine learning techniques are presented along with a *restricted Boltzmann machine (RBM)* for *generative modeling* on *Fashion-MNIST* using Python. This work was conducted at the context of my thesis preparation in Electrical & Computer Engineer at Technical University of Crete (TUC), Greece.

![alt text](https://github.com/dkomni/machine-learning-overview/blob/master/deep_nn.png?raw=true | width=100)

The pdf file *Machine-Learning-Overview-DKomninos* included in this repository presents some general Artificial Intelligence (AI) and Machine Learning (ML) concepts with thorough mathematics. Several basics such as *neural networks*, *gradient descent* and the *backpropagation algorithm* are discussed, along with comments on *hyperparameters*, *datasets* and *training*. Also, a general idea about *generative adversarial neural networks (GANs)* is introduced, arriving at the investigation of *restricted Boltzmann Machines (RBMs)* for generative modeling. A RBM is trained on the Fashion-MNIST dataset using the *Contrastive Divergence* algorithm and learns to generate synthetic data (reconstruct samples) that resemble the ones of the Fashion-MNIST dataset.

### Notes
Whether using Windows, macOS or Linux, it is recommended that you create a virtual environment for Python and install the required packages within it. In this way, you isolate your operating system from the new installations and prevent possible compatibility issues. Also, make sure that you install the Jupyter kernel, which can be used to run Jupyter commands inside the virtual environment.

Special thanks to my friend Antonios Kastellakis for providing the code on RBMs and helped me with my introduction on generative modeling. You may find his thesis 'Analog and Digital Quantum Neural Networks: Basic Concepts and Applications', where he trains a RBM on the MNIST dataset of hand-written digits at: http://dimitrisangelakis.org/theses/

### Some insight on the notebooks

#### 01_MachineLearning_Overview
Implementation of simple neural networks, backpropagation algorithm and some data fitting using pure Python.

#### 02_MachineLearning_Keras
Implementation of neural networks and tests on hyperparameters using Keras package from Tensorflow.

#### 03_RBMs
Implementation of a restricted Boltzmann machine (RBM) for generative modeling on the Fashion-MNIST dataset, trained with Contrastive Divergence algorithm and reconstructing data with Gibbs sampling, along with results and comments.

*Python version*: 3.10.6

| Package    | Version |
| ---------- | ------- |
| Tensorflow | 2.11.0  |
| Matplotlib | 3.6.3   |
| Numpy      | 1.24.2  |
| ipykernel  | 6.21.1  |
| silence-tensorflow | 1.2.1 |
