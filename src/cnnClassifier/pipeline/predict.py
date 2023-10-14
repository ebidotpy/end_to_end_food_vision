import numpy as np
import tensorflow as tf
import os


class_names = ['chicken_curry',
 'chicken_wings',
 'fried_rice',
 'grilled_salmon',
 'hamburger',
 'ice_cream',
 'pizza',
 'ramen',
 'steak',
 'sushi']
class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
    
    def predict(self):
        model  = tf.keras.models.load_model(os.path.join("artifacts", "training", "model.h5"))
        imageName = self.filename
        test_image = tf.keras.preprocessing.image.load_img(imageName, target_size=(224, 224))
        test_image = tf.keras.preprocessing.image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        pred = np.argmax(model.predict(test_image), axis=1)
        pred = np.ndarray.tolist(pred)
        pred_class = class_names[pred[0]]
        return [{"image": pred_class}]

# pred = PredictionPipeline("artifacts/data_ingestion/10_food_classes_10_percent/test/chicken_curry/838.jpg")
# pred.predict()