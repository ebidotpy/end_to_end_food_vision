from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
from cnnClassifier.pipeline.stage_03_training import TrainingPipeline
from cnnClassifier.pipeline.stage_04_evaluation import Evaluation

STAGE_NAME = "Data Ingestion"
try:
    logger.info(f">>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx=======================x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Prepare Callbacks"
try:
    logger.info(f">>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<")
    obj = PrepareBaseModelPipeline()
    obj.main()
    logger.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx=======================x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Training"
try:
    logger.info(f">>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<")
    obj = TrainingPipeline()
    obj.main()
    logger.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx=======================x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Evaluation"
try:
    logger.info(f">>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<")
    obj = PrepareBaseModelPipeline()
    obj.main()
    logger.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx=======================x")
except Exception as e:
    logger.exception(e)
    raise e