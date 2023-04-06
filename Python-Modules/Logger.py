import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

class Logger2:
    def __init__(self, name):
        self.logger = logging.getLogger(name)

    def info(self, message, *args, **kwargs):
        self.logger.info(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self.logger.warn(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)

    def log_separator(self):
        self.logger.info("---------------------")


class Logger:
    def __init__(self, filepath, is_continue_execution):
        self._filepath = filepath

        if is_continue_execution:
            self._logfile = open(self._filepath, "a+")
        else:
            self._logfile = open(self._filepath, "w+")

    def log(self, log_message):
        self._logfile.write(log_message)
        self._logfile.write("\n")
        self._logfile.flush()

    def log_separator(self):
        self.log("---------------------")

    '''
        The "getstate" and "setstate" functions are for pickling -
        we use multiprocessing process pool, which serialize objects using pickle.
        pickle serializes the content that returns from "getstate" method,
        and using "setstate" method when deserializing.
        file handle cannot be serialized - so we have to remove it from the serialized
        content, and add re-create it when de-serializing.
    '''
    def __getstate__(self):
        odict = self.__dict__.copy() # copy the dict since we change it
        del odict['_logfile']              # remove filehandle entry
        return odict

    def __setstate__(self, dict):
        self._filepath = dict['_filepath']
        self._logfile = open(self._filepath, "a+")
        self.__dict__.update(dict)   # update attributes

