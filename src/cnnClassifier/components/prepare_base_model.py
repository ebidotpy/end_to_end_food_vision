import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


    def get_base_model(self):
        self.model = tf.keras.applications.EfficientNetB0(include_top=self.config.include_top)

        self.save_model(self.config.base_model_path, self.model)
    @staticmethod
    def _prepare_full_model(base_model, input_shape, freeze_all, freeze_till, classes, learning_rate):
        if freeze_all:
            base_model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in base_model.layers[:-freeze_till]:
                base_model.trainable = False
        
        inputs = layers.Input(shape=input_shape, name="input_layer")
        
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
        x = layers.Dense(classes)(x)

        outputs = layers.Activation("softmax", dtype=tf.float32, name="siftmax_float32")(x)
        full_model = tf.keras.Model(inputs, outputs)

        full_model.compile(loss="categorical_crossentropy", 
                           optimizer=tf.keras.optimizers.Adam(), 
                           metrics=["accuracy"])
        
        full_model.summary()
        return full_model
        
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            base_model=self.model, 
            input_shape=self.config.input_shape, 
            freeze_all=self.config.freeze_all, 
            freeze_till=self.config.freeze_till, 
            classes=self.config.classes, 
            learning_rate=self.config.learning_rate
        )

        self.save_model(self.config.updated_base_model_path, self.full_model)