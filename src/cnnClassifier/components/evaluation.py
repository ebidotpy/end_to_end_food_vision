import tensorflow as tf
from cnnClassifier.utils.common import save_json
from cnnClassifier.entity.config_entity import EvaluationConfig
from pathlib import Path

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.model_path
        )
    
    def evaluate_model(self, test_data):
        self.score = self.model.evaluate(test_data)

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path(self.config.score_path), data=scores)