import unittest

from src.run_naming import make_run_name


class RunNamingTests(unittest.TestCase):
    def test_custom_label_is_readable_and_unique(self):
        first = make_run_name("baseline yolo26")
        second = make_run_name("baseline yolo26")
        self.assertTrue(first.startswith("baseline-yolo26-"))
        self.assertNotEqual(first, second)

    def test_default_prefix_and_path_separator_sanitization(self):
        self.assertTrue(make_run_name().startswith("yolo26-"))
        self.assertTrue(make_run_name("trial/one").startswith("trial-one-"))

    def test_rejects_label_without_usable_characters(self):
        with self.assertRaises(ValueError):
            make_run_name("/\\")


if __name__ == "__main__":
    unittest.main()
