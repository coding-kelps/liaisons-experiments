import logging
import tqdm

# https://stackoverflow.com/a/38739634
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)  

def setup_logger():
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    log.addHandler(TqdmLoggingHandler())
