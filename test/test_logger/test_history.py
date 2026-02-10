from toolkit.utils.logger.training_logger import History
from utils.test_base import TestBase


"""Test logging the items adds to the history correctly"""


class TestLoggerHistory(TestBase):

    def test_history(self):
        history = History()

        # Log the items
        history.write("train", "Pos", 0)
        history.write("train", "Pos", 0.1)
        history.write("train", "Pos", 0.2)
        history.write("train", "Vel", 1)
        history.write("train", "Vel", 1.1)
        history.write("train", "Vel", 1.2)
        history.write("val", "Pos", 2)
        history.write("val", "Pos", 2.1)
        history.write("val", "Pos", 2.2)
        history.write("val", "Vel", 3)
        history.write("val", "Vel", 3.1)
        history.write("val", "Vel", 3.2)

        # Evaluate the logged items
        self.assertListEqual([0, 0.1, 0.2], history.training['Pos'])
        self.assertListEqual([1, 1.1, 1.2], history.training['Vel'])
        self.assertListEqual([2, 2.1, 2.2], history.validation['Pos'])
        self.assertListEqual([3, 3.1, 3.2], history.validation['Vel'])
