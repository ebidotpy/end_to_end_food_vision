from cnnClassifier import logger
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_callbacks import PrepareCallbacks 
from cnnClassifier.components.training import Training
from cnnClassifier.components.data_loader import DataLoader

STAGE_NAME = "Training"

class TrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        # prepare callbacks
        prepare_callbacks_config = config.get_prepare_callbacks_config()
        prepare_callbacks = PrepareCallbacks(config=prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()

        # data loader
        data_loader_config = config.get_data_loader_config()
        data_loader = DataLoader(config=data_loader_config)
        data_loader.load_data()
        train, test = data_loader.prepare_data()

        # training
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.training(
            callbacks_list=callback_list, 
            train_data=train, 
            test_data=test
        )


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx=======================x")
    except Exception as e:
        logger.exception(e)
        raise e