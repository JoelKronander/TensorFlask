# -*- coding: utf-8 -*-
"""MNIST digit classification using pretrained Tensorflow/layer models

The main class of the module is MNISTClassifier
MNISTClassifier.classify(request_list) parses JSON formatted requests and
returns a JSON formatted response containing the classifications.
"""
import base64
import io
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from model.cnn_model import cnn_model_graph

class MNISTClassifier:
    """MNISTClassifier initialized a tensorflow session with a convolutional MNIST
    digit classification network with loaded pretrained weights. The class assume
    pretrained weights is available and only handles inference."""

    def __init__(self, model_param_path='model/model_params.npz', max_batch_size=64):
        """Initalized tensorflow session, set up tensorflow graph, and loads
        pretrained model weights

        Arguments:
            model_params (string) : Path to pretrained model parameters
            max_batch_size (int) : Maximum number of images handled in a single
                                   call to classify.
        """
        self.max_batch_size = max_batch_size
        self.model_param_path = model_param_path

        #Setup tensorflow graph and initizalize variables
        self.session = tf.InteractiveSession()
        self.x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x')
        self.network, self.predictions, self.probabilities = self._load_model_definition()
        self.session.run( tf.global_variables_initializer() )

        #Load saved parametes
        self._load_model_parameters()

    def _load_model_definition(self):
        """Loads the model graph used for classification, subclasses can
        overload this function to use other model architectures

        Returns:
            Tuple consisting of
            (TensorLayer Layer corresponding to the last layer in the model specification,
             TensorFlow node evaluating to class predictions as tensor of size [batch_size,1],
             TensorFlow node evaluating to the class probabilites as tensor of size [batch_size,10]
             )
        """
        network = cnn_model_graph(self.x)
        probs = tf.nn.softmax(network.outputs)
        preds = tf.argmax(probs,1)
        return (network, preds, probs)

    def _load_model_parameters(self):
        """Loads the model parameters from specified path, subclasses can
        overload this function to load other models and model parameters
        """
        load_params = tl.files.load_npz(path='', name=self.model_param_path)
        tl.files.assign_params(self.session, load_params, self.network)

    def classify(self, requests_list):
        """Takes a list of dictonaries conatining image specifications
            and returns list of dicttionaries containing classifications.

            The maximum number of images that can be handled in the requests_list
            is specificed by the set MNISTClassifier.max_batch_size.

        Arguments:
            requests_list : List of dictionaries, each of the form :
                {'image' : base64_encoded_image_string} where base64_encoded_image_string
                is base64 encoded raw 28 by 28 graysacale image data (e.g. read from file)

        Exceptions:
            MNISTClassifierInputError : if the specified input is invalid or the
                number of images in the request_list is greater than
                MNISTClassifier.max_batch_size.

        Returns:
            classifications : List of dictionaries, one for each classified image,
                each of the form:
                {'class' :  classification, 'prob' : probability}
                where classification is the infered class 0-9 and probability is
                the probability of the infered class.
        """

        if len(requests_list) > self.max_batch_size:
            raise MNISTClassifierInputError(
                'The maximum number of images can be processed in a single POST request is {}'
                .format(self.max_batch_size))

        #create batch of normalized images in float32 with shape [num_images, 28,28,1]
        images = np.zeros((len(requests_list), 28, 28, 1), dtype=np.float32)
        for i, request in enumerate(requests_list):
            image = conv_b64_to_img(request['image'])
            self.check_input(image)
            images[i] = np.float32(image.reshape(28,28,1))/256.0

        classifications = []

        dp_dict = tl.utils.dict_to_one( self.network.all_drop )    # disable dropout layers
        feed_dict={self.x : images}
        feed_dict.update(dp_dict)
        pred, prob = self.session.run([self.predictions, self.probabilities], feed_dict=feed_dict)
        for i in range(pred.shape[0]):
            classifications.append({'class' : int(pred[i]), 'probability' : float(prob[i][pred[i]])})
        return classifications

    def check_input(self, image):
        """Checks if the numpy array image is valid as input into tensorflow
         graph, raises an MNISTClassifierInputError if not"""
        if not image.shape == (28,28):
            raise MNISTClassifierInputError("""Input images should be grayscale 28 by 28""")


class MNISTClassifierError(Exception):
    """Base class for Exceptions raised by MNISTClassifier"""


class MNISTClassifierInputError(MNISTClassifierError):
    """Exception raised for errors in the input."""


def conv_b64_to_img(base64_enc_image):
    """Converts a base64 encoded image to a numpy array"""
    sbuf = io.BytesIO(base64.b64decode(base64_enc_image))
    pimg = Image.open(sbuf)
    return np.array(pimg)
