from cgi import test
from email import message
from locale import LC_ALL

from matplotlib.pyplot import text
import translator
import pytest
from unittest.mock import call
import pandas as pd

t = translator.Translator()


class TestTranslate(object):
    def test_blank_string(self):
        test_argument = ""
        expected = "I'm sorry."
        actual = t.translate(text=test_argument)
        message = "Wrong blank translation, expected is {0}, but given {1}".format(
            expected, actual
        )
        assert isinstance(actual, str)
        assert actual == expected, message

    def test_none_input(self):
        test_argument = None
        with pytest.raises(ValueError):
            t.translate(text=test_argument)

    def test_normal_sentence(self):
        test_argument = "adsa"
        expected = "Adsa"
        actual = t.translate(text=test_argument)
        message = "Wrong translation, expected is {0}, but given {1}".format(
            expected, actual
        )
        assert isinstance(actual, str)
        assert actual == expected, message

    def test_type_error(self):
        test_argument = 9
        with pytest.raises(ValueError):
            t.translate(text=test_argument)

    def test_normal_sentence_2(self):
        test_argument = "加藤産業㈱阪神支店\u3000ﾘﾍﾞｰﾄ3月"
        expected = "- I'll be right back. - I'll be right back. - I'll be right back. I'll be right back."
        actual = t.translate(text=test_argument)
        message = "Wrong translation, expected is {0}, but given {1}".format(
            expected, actual
        )
        assert isinstance(actual, str)
        assert actual == expected, message


class TestColumnTranslate(object):
    def translate_bug_free(self):
        return_values = {
            "adsa": "Adsa",
            "加藤産業㈱阪神支店\u3000ﾘﾍﾞｰﾄ3月": "- I'll be right back. - I'll be right back. - I'll be right back. I'll be right back.",
        }
        return return_values[row]

    def test_normal_column_translation(self, mocker):
        mydata = {
            "text": ["adsa", "加藤産業㈱阪神支店\u3000ﾘﾍﾞｰﾄ3月"],
            "translation": [
                "Adsa",
                "- I'll be right back. - I'll be right back. - I'll be right back. I'll be right back.",
            ],
        }
        mydata = pd.DataFrame(data=mydata)
        translate_mock = mocker.patch(
            "translator.Translator.translate", side_effect=self.translate_bug_free
        )
        t.column_translate(dataframe=mydata, colum_name="text")
        assert translate_mock.call_args_list == [
            call("adsa"),
            call("加藤産業㈱阪神支店\u3000ﾘﾍﾞｰﾄ3月"),
        ]

