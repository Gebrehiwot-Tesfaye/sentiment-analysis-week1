import unittest
import pandas as pd
from src.utils.data_utils import preprocess_text

class TestDataUtils(unittest.TestCase):
    def test_preprocess_text(self):
        test_text = "Sample text"
        processed_text = preprocess_text(test_text)
        self.assertIsInstance(processed_text, str)

if __name__ == '__main__':
    unittest.main() 