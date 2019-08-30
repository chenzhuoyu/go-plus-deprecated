import unittest

from goplus.types import Type

class TestInferrer(unittest.TestCase):
    def test_inferrer(self):
        print(Type.UntypedInt)

if __name__ == '__main__':
    unittest.main()
