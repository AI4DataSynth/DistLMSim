"""Test runner: discovers and runs all tests."""

import sys
import unittest

sys.path.insert(0, '.')

loader = unittest.TestLoader()
suite = loader.discover('tests', pattern='test_*.py')
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
