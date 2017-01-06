# -*- coding: utf-8 -*-
""" Flask webservice application accepting HTTP POST request with JSON data
corresponding to a list of MNIST image to be classified.
See https://github.com/JoelKronander/TensorFlask for a detailed specification of
the JSON data excepted for the request.

Example:
    The webservice can for example be deployed using gunicorn:
    $gunicorn server:app
"""
from flask import Flask, request, jsonify
from mnist_classifiers import MNISTClassifier, MNISTClassifierInputError

#The maximum number of images to handle in a single request
#Should be set to reflect the memory and computational resources of the
#machine the server is running on.
MAX_BATCH_SIZE = 64

#Create an instance of a MNISTClassifier that initizalizes and controls the
#state of the tensorflow graph
REQUEST_HANDLER = MNISTClassifier(max_batch_size=MAX_BATCH_SIZE)

#Create the Flask application handling HTTP requests
FLSK_APP = Flask(__name__)

@FLSK_APP.route('/mnist/classify', methods=['POST'])
def classify_mnist_images():
    """Unpacks the JSON data passed with the POST request and forwards it to the
    MNISTClassifier for classification"""
    if request.method == 'POST':
        resp = jsonify([])
        try:
            classifications = REQUEST_HANDLER.classify(request.json['requests'])
            data = {
                'responses'  : classifications,
            }
            resp = jsonify(data)
            resp.status_code = 200
            return resp
        except MNISTClassifierInputError as excep:
            resp = bad_input("Invalid input detected: {}"
                             .format(excep))
            return resp
        #Handle Internal Server Errors
        except Exception as excep:
            resp = bad_input("Unexpected server API error: {}"
                             .format(excep))

def bad_input(message):
    """Returns a 404 status code JSON response with the provided message"""
    response = jsonify({'message': message})
    response.status_code = 404
    return response
