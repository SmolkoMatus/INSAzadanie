import logging
from sklearn.pipeline import Pipeline

from model import config, manager, inputers

logger = logging.getLogger(__name__)
__version__ = manager.get_version(config.PATH_TO_VERSION)

pipeline = Pipeline([
  ('categorical_transformer', inputers.CategoricalTransformer(config.CATEGORICAL_FEATURES)),
  ('numerical_transformer', inputers.NumericalTransformer(config.NUMERICAL_FEATURES)),
  ("classifier_model", config.CLASSIFIER),
])