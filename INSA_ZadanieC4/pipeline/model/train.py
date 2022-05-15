import logging

import config, manager
from pipeline import pipeline, __version__

logger = logging.getLogger(__name__)


def run():
  logger.info('Running train.py file.')
  logger.info(f'Loading dataset from: {config.PATH_TO_TRAIN_DATASET}')
  dataset = manager.load_dataset(config.PATH_TO_TRAIN_DATASET)
  x_train, y_train = dataset[config.FEATURES], dataset[config.TARGET]
  logger.info(f'Creating new pipeline.')
  logger.info(f'Training pipeline...')
  pipeline.fit(x_train, y_train)

  manager.save_pipeline(config.PATH_TO_MODEL, config.MODEL_NAME, __version__, pipeline)
  logger.info(f'Done!')


if __name__ == "__main__":
  run()
