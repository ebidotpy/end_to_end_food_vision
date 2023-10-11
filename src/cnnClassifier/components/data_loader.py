from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cnnClassifier.entity.config_entity import DataLoaderConfig

class DataLoader:
    def __init__(self, config: DataLoaderConfig):
        self.config = config
    
    def load_data(self):
        self.train_dir = self.config.train_dir
        self.test_dir = self.config.test_dir

    def prepare_data(self):
        train_datagen = ImageDataGenerator(rescale=1/255.)
        test_datagen = ImageDataGenerator(rescale=1/255.)

        train_data_10_percent = train_datagen.flow_from_directory(self.train_dir, 
                                                                  target_size=self.config.image_shape, 
                                                                  batch_size=self.config.batch_size, 
                                                                  class_mode=self.config.class_mode)
        test_data = train_datagen.flow_from_directory(self.test_dir, 
                                                      target_size=self.config.image_shape, 
                                                      batch_size=self.config.batch_size, 
                                                      class_mode=self.config.class_mode)
        return train_data_10_percent, test_data