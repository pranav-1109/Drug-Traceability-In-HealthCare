from django.test import TestCase

class MyTestCase(TestCase):
    def test_example(self):
        # Test your code here
        self.assertEqual(2 + 2, 4)