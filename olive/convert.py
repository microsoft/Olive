import traceback
from .conversion.base_converter import BaseConverter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert(conversion_config):
    converter = BaseConverter.create_converter(conversion_config)
    converter.convert()
    logger.info("Conversion succeeded")