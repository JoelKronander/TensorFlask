#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""A simple client demonstrating how to send requests to the TensorFlask
MNIST digit classification webservice. https://github.com/JoelKronander/TensorFlask

The client takes as input a path to a folder containing images to be classfied
and then constructs a corresponding JSON query object for the images. The JSON
query object is then sent via a HTTP POST request to the TensorFlask
classification webservice. The JSON response from the server with the
corresponding classification is then parsed and printed.

Arguments:
    --download_mnist:Download 10 random MNIST test images to mnistimages/ before
                     loading images from the specified folderpath
    --folderpath=(string):Path to folder of images to be classfied, images
                          should be grayscale (*.png) with a resolution of 28x28
    --server=(string):URL to webserver
                      [default value="http://127.0.0.1:8000/"]

Example:
    Dowload 10 random MNIST test images and get classifications of these using
    the webserver running on http://127.0.0.1:8000/.

    $python3 test_client.py --download_mnist --folderpath="mnistimages/"
    --server="http://127.0.0.1:8000/"

    Classify the images in the mnistimages/ folder using the webserver running
    on http://127.0.0.1:8000/.

    $python3 test_client.py --folderpath="mnistimages/" --server="http://127.0.0.1:8000/"
"""
import os
import argparse
import requests
import base64
import glob
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy.misc import imsave

def main():
    #Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_mnist",
                        help="Dowload 10 random MNIST test images to minstimages/",
                        action="store_true")
    parser.add_argument("--folderpath",
                        default='mnistimages/',
                        help="Path to folder of 28 by 28 grayscale .png images to be classfied",
                        type=str)
    parser.add_argument("--server",
                        default='http://127.0.0.1:8000/',
                        help="URL to webserver",
                        type=str)
    args = parser.parse_args()

    if(args.download_mnist):
        #Dowload MNIST data using tensorflow utils and save 10 random images
        #from test set to mnistimages/
        if not os.path.exists('mnistimages/'):
            os.makedirs('mnistimages')
        mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
        rand_indicies = np.random.randint(mnist.test.images.shape[0], size=10)
        test_images = mnist.test.images[rand_indicies]
        test_labels = mnist.test.labels[rand_indicies]
        for i, test_image in enumerate(test_images):
            imsave('mnistimages/img_{0:02d}_truelabel_{1:1d}.png'.format(i, int(test_labels[i])), test_image.reshape(28,28))

    #Look for files in folderpath and populate json request object
    requests_list = []
    filenames = glob.glob(args.folderpath + '/*.png')
    for filename in filenames:
        with open(filename, 'rb') as image_file:
            image_json_obj = {
                'image': base64.b64encode(image_file.read()).decode('UTF-8')
            }
        requests_list.append(image_json_obj)

    #Send request to server
    print("Requesting classifications for {} images...".format(len(requests_list)))
    response = requests.post(args.server+'/mnist/classify', json={'requests' : requests_list})

    #Parse JSON response from server
    json_response = response.json()
    print('JSON response from server :')
    print('    ',json_response,'\n')
    if(response.status_code == 200):
        print('Classifications returned from server :')
        for filename, response in zip(filenames, json_response['responses']):
            print('    Image {} classified as a {} with approximate probability/score {}'
                  .format(filename, response['class'], response['probability']))

if __name__ == '__main__':
    main()
