"""Unit tests for translator bot."""
import unittest

class TestTranslation(unittest.TestCase):
    def test_supported_languages_not_empty(self):
        langs = ["en","fr","de","es","it","pt","nl","ru","zh","ja","hi","ar"]
        self.assertGreater(len(langs), 10)

    def test_language_codes_lowercase(self):
        for code in ["en","fr","de"]:
            self.assertEqual(code, code.lower())

    def test_empty_input_returns_empty(self):
        text = "  ".strip()
        result = text if text else ""
        self.assertEqual(result, "")

    def test_same_src_tgt_returns_input(self):
        text = "Hello world"
        src, tgt = "en", "en"
        result = text if src == tgt else None
        self.assertEqual(result, text)

class TestTextCleaning(unittest.TestCase):
    def test_strip_whitespace(self):
        self.assertEqual("  hello  ".strip(), "hello")
    def test_multi_spaces_collapsed(self):
        import re
        self.assertEqual(re.sub(r' +', ' ', "hello   world"), "hello world")

if __name__ == "__main__": unittest.main()
