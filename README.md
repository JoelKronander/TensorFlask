## TensorFlask
A simple web service API classifying MNIST digits from HTTP POST requests built using [Flask](http://flask.pocoo.org/), [TensorFlow](https://www.tensorflow.org/) and [TensorLayer](http://tensorlayer.readthedocs.io/en/latest/)

The API uses HTTP POST operations to classify images of handwritten [MNIST digits](http://yann.lecun.com/exdb/mnist/) that is sent in the request. The single POST request available is /mnist/classify.
The API uses JSON for both requests and responses, see below for a detailed specification of the JSON data format.
Currently the API only supports 28 by 28 grayscale images and only handles a set maximum batch size of images in each
request (see below for how to set the max batch size).

## JSON request format
The HTTP POST request /mnist/classify expects a JSON request. Example JSON data for the request:
```python
{
  "requests":[
      {
        "image":"/9j/7QBEUGhvdG9...image contents...eYxxxzj/Coa6Bax//Z"
      },
  ]
}
```
* requests - A list of requests, one for each image
    * image - The image data for this request provided as base64-encoded raw image data. The API only accepts 28 by 28 pixel grayscale images at the moment.

## JSON response format
```python
{
  "responses":[
      {
        "class":1,
        "probability":0.98
      },
  ]
}
```
* responses - A list of responses, one for each image
    * class - The type of digit the image represents [0-9]
    * probability - The inferred probability of the predicted class [0-1] (Softmax score)

## Installing requirements
The API uses python3 and the requirements can be installed by

```bash
    $pip3 install -r requirements.txt
```

## Running the server
The Flask application can be deployed using e.g. gunicorn using:

```bash
    $gunicorn server_application:app
```

## Running the simple test client
After starting the server requests can be sent using the test client.
For detailed use of the test client see:

```bash
    $python3 simple_client.py --help
```

Example use, downloading 10 MNIST images and submitting them as a request for
classification to a local server:

```bash
    $python3 simple_client.py --download_mnist --server=http://127.0.0.1:8000/
```

## Training new model parameters
For convinience pretrained model parameters for the convolutional neural network
is supplied in the model/ directory. However, the model can also be retrained by
running the train_model.py script in the model directory.

```bash
    $python3 python3 train_model.py
```
