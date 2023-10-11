from cnnClassifier.constants import *
import os
from pathlib import Path
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig, 
                                                DataLoaderConfig, 
                                                PrepareBaseModelConfig, 
                                                PrepareCallbacksConfig, 
                                                TrainingConfig, 
                                                EvaluationConfig)


class ConfigurationManager:
    def __init__(
            self, 
            config_filepath = CONFIG_FILE_PATH, 
            params_filepath = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir, 
            dataset_url=config.dataset_url,
            local_data_file=config.local_data_file, 
            unzip_dir=config.unzip_dir
            
        )
        return data_ingestion_config
    
    def get_data_loader_config(self) -> DataLoaderConfig:
        config = self.config.data_loader
        params = self.params

        data_loader_config = DataLoaderConfig(
            root_dir=config.root_dir,
            train_dir=config.train_dir, 
            test_dir=config.test_dir, 
            image_shape=params.IMAGE_SHAPE, 
            batch_size=params.BATCH_SIZE, 
            class_mode=params.CLASS_MODE
        )

        return data_loader_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        params = self.params

        prepare_base_model = PrepareBaseModelConfig(
            root_dir=config.root_dir, 
            base_model_path=config.base_model_path, 
            updated_base_model_path=config.updated_base_model_path, 
            include_top=params.INCLUDE_TOP, 
            trainable=params.TRAINABLE, 
            input_shape=params.INPUT_SHAPE, 
            freeze_all=params.FREEZE_ALL, 
            freeze_till=params.FREEZE_TILL, 
            learning_rate=params.LEARNING_RATE, 
            classes=params.CLASSES
        )

        return prepare_base_model
    
    def get_prepare_callbacks_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks

        prepare_callbacks_config = PrepareCallbacksConfig(
            root_dir=config.root_dir, 
            tensorboard_root_log_dir=config.tensorboard_root_log_dir, 
            checkpoint_model_filepath=config.checkpoint_model_filepath
        )

        return prepare_callbacks_config
    
    def get_training_config(self):
        training = self.config.training
        params = self.params
        prepare_base_model = self.config.prepare_base_model
        data_loader = self.config.data_loader


        create_directories([training.root_dir])

        training_config = TrainingConfig(
            root_dir=training.root_dir, 
            trained_model_path=training.trained_model_path, 
            updated_base_model_path=prepare_base_model.updated_base_model_path, 
            train_dir=data_loader.train_dir, 
            test_dir=data_loader.test_dir, 
            epochs=params.EPOCHS
        )

        return training_config
    
    def get_evaluation_config(self):
        config = self.config.evaluation

        evaluation = EvaluationConfig(
            model_path=config.model_path, 
            score_path=config.score_path
        )
        return evaluation