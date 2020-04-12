from flask import Flask, request, jsonify
import keras
import numpy as np
from keras.applications import vgg16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions

#Load  the VGG model
vgg_model = vgg16.VGG16(weights='imagenet')
vgg_model._make_predict_function() 

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    label = None
    # parameters = request.get_json(force=True)
    # f1 = int(parameters['f1'])
    # f2 = int(parameters['f2'])
    # f3 = int(parameters['f3'])
    # spending_score = model.predict(([f1, f2, f3],))[0]
    # print(spending_score)
    # return jsonify(spending_score)

    # filename = 'images/cat.jpg'
    filename = request.files['file']
    # load an image in PIL format
    original = load_img(filename, target_size=(224, 224))
    print('PIL image size',original.size)

    # convert the PIL image to a numpy array
    # IN PIL - image is in (width, height, channel)
    # In Numpy - image is in (height, width, channel)
    numpy_image = img_to_array(original)
    print('numpy array size',numpy_image.shape)

    # Convert the image / images into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    image_batch = np.expand_dims(numpy_image, axis=0)
    print('image batch size', image_batch.shape)

    # prepare the image for the VGG model
    processed_image = vgg16.preprocess_input(image_batch.copy())

    # get the predicted probabilities for each class
    predictions = vgg_model.predict(processed_image)
    #print(predictions)
    # convert the probabilities to class labels
    # We will get top 5 predictions which is the default
    label = decode_predictions(predictions)
    print(label)
    return jsonify(label[0][0][1])

if __name__ == '__main__':
    app.run(host='0.0.0.0')