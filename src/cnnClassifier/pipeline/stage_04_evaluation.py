from cnnClassifier import logger
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_loader import DataLoader
from cnnClassifier.components.evaluation import Evaluation

STAGE_NAME = "Evaluation"

class PrepareBaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        
        data_loader_config = config.get_data_loader_config()
        data_loader = DataLoader(config=data_loader_config)
        data_loader.load_data()
        _, test = data_loader.prepare_data()


        evaluation_config = config.get_evaluation_config()
        evaluation = Evaluation(config=evaluation_config)
        evaluation.get_base_model()
        evaluation.evaluate_model(test)
        evaluation.save_score()
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<")
        obj = PrepareBaseModelPipeline()
        obj.main()
        logger.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx=======================x")
    except Exception as e:
        logger.exception(e)
        raise e