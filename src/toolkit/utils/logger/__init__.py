import time

from .training_logger import Logger


class CrudLogger:

    @staticmethod
    def _log(prefix, message, indent):
        tab = "  " * indent
        print(f"[{prefix}] {tab}{message}")

        return time.time()

    def error(self, message, indent=0):
        return self._log('ERROR', message, indent)

    def info(self, message, indent=0):
        return self._log('INFO', message, indent)


log = CrudLogger()


def log_time(message: str, pre=None, indent=0):
    def decorator(function):
        def wrapper(*args, **kwargs):
            _message = message

            start = time.time()

            if pre:
                log.info(pre, indent=indent)

            value = function(*args, **kwargs)

            if "%LEN%" in _message:
                _message = _message.replace("%LEN%", str(len(value)))

            log.info(f"{_message} in {time.time() - start:.4f}s.", indent=indent)

            return value
        return wrapper
    return decorator
