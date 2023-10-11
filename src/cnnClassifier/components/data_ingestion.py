import urllib.request as request
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from zipfile import ZipFile
from pathlib import Path
from cnnClassifier.entity.config_entity import DataIngestionConfig
import os


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_dataset(self):
        if not os.path.exists(self.config.local_data_file):
            # filename = wget.download(self.config.dataste_url, out=self.config.local_data_file)
            filename, headers = request.urlretrieve(
                url = self.config.dataset_url,
                filename = self.config.local_data_file
            )    
            logger.info(f"{filename} download!")
        
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")
        
    
    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        with ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)