try:
    import absl.logging
except ImportError:
    pass
else:
    absl.logging.set_verbosity("info")
    absl.logging.set_stderrthreshold("info")
    absl.logging._warn_preinit_stderr = False

import logging
import tensorflow as tf
import os

logger = logging.getLogger(__name__)

from .marker_utils import get_marker
from .processor_utils import PassageHandle, DocumentHandle, DocumentSplitterHandle
from .msmarco_documents import  MsMarcoDocumentProcessor
from .msmarco_passages import  MsMarcoPassageProcessor
from .robust04 import Robust04Processor
from .cord19 import Cord19Processor