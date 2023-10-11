from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path
import tensorflow as tf


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    
    
    def training(self, callbacks_list: list, train_data, test_data):
        self.history = self.model.fit(
            train_data,
            epochs=self.config.epochs, 
            steps_per_epoch=len(train_data), 
            validation_data=test_data, 
            validation_steps=int(0.15 * len(test_data)), 
            callbacks=callbacks_list
        )

        self.save_model(
            path=self.config.trained_model_path, 
            model=self.model
        )