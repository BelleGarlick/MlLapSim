from toolkit.utils.logger import Logger
from utils.test_base import TestBase
from unittest.mock import patch

"""Test logged functionality"""


class TestLogging(TestBase):

    @patch('builtins.print')
    def test_print_statements(self, mocked_print):
        logger = Logger(labels=['P', 'V'], log_every=1, n_partitions=2)

        logger.write(0, 0, 3, [3, 4], partition=0)
        logger.write(0, 1, 3, [3.1, 4.2], partition=0)
        logger.write(0, 2, 3, [2.6, 4.1], partition=0)

        logger.write(0, 0, 3, [2.9, 3.9], partition=1)
        logger.write(0, 1, 3, [3, 4.1], partition=1)
        logger.write(0, 2, 3, [2.5, 4], partition=1)

        logger.write_val(0, 0, 2, [3.6, 4.6])
        logger.write_val(0, 0, 2, [3.4, 4.6])

        logger.flush(0)

        self.assertIn("Partition: 1/2", mocked_print.mock_calls[0].args[0])
        self.assertIn("Batch: 2/3", mocked_print.mock_calls[1].args[0])
        self.assertIn("P: 2.92", mocked_print.mock_calls[4].args[0])
        self.assertIn("V: 4.05", mocked_print.mock_calls[5].args[0])
        self.assertIn("New best val loss: 8.1", mocked_print.mock_calls[9].args[0])

    @patch('builtins.print')
    def test_partitions_not_in_when_1_partition(self, mocked_print):
        logger = Logger(log_every=1, n_partitions=1)

        logger.write(0, 0, 3, [3])

        logger.flush(0)

        self.assertNotIn("Partition", mocked_print.mock_calls[0].args[0])

    def test_logged_data(self):
        """Test logging data adds to history and is obtained correct and callback works"""
        global best_reached
        best_reached = 0

        def inc_best_reached(_):
            global best_reached
            best_reached += 1

        logger = Logger(labels=['T1', 'T2'])
        logger.set_best_val_loss_callback(inc_best_reached)

        # Test first epoch with dummy values
        logger.write(0, 0, 3, [3, 4])
        logger.write(0, 1, 3, [3.1, 4.2])
        logger.write(0, 2, 3, [2.6, 4.1])

        logger.write_val(0, 0, 2, [3.6, 4.6])
        logger.write_val(0, 0, 2, [3.4, 4.6])

        logger.flush(0)

        self.assertEqual(1, best_reached)
        self.assertFloatListEqual([2.9], logger.history.training['T1'])
        self.assertFloatListEqual([4.1], logger.history.training['T2'])
        self.assertFloatListEqual([3.5], logger.history.validation['T1'])
        self.assertFloatListEqual([4.6], logger.history.validation['T2'])

        # Test second epoch with dummy values
        logger.write(1, 0, 3, [2.8, 3.8])
        logger.write(1, 1, 3, [2.9, 4])
        logger.write(1, 2, 3, [2.4, 3.9])

        logger.write_val(1, 0, 2, [3.6, 4.8])
        logger.write_val(1, 0, 2, [3.6, 4.6])

        logger.flush(0)

        self.assertEqual(1, best_reached)  # no new best, so not updated
        self.assertFloatListEqual([2.9, 2.7], logger.history.training['T1'])
        self.assertFloatListEqual([4.1, 3.9], logger.history.training['T2'])
        self.assertFloatListEqual([3.5, 3.6], logger.history.validation['T1'])
        self.assertFloatListEqual([4.6, 4.7], logger.history.validation['T2'])

        # Test third epoch with dummy values
        logger.write(1, 0, 3, [2.7, 3.7])
        logger.write(1, 1, 3, [2.8, 3.9])
        logger.write(1, 2, 3, [2.3, 3.8])

        logger.write_val(1, 0, 2, [3.2, 4.3])
        logger.write_val(1, 0, 2, [3.0, 4.3])

        logger.flush(0)

        self.assertEqual(2, best_reached)  # no new best, so not updated
        self.assertFloatListEqual([2.9, 2.7, 2.6], logger.history.training['T1'])
        self.assertFloatListEqual([4.1, 3.9, 3.8], logger.history.training['T2'])
        self.assertFloatListEqual([3.5, 3.6, 3.1], logger.history.validation['T1'])
        self.assertFloatListEqual([4.6, 4.7, 4.3], logger.history.validation['T2'])
