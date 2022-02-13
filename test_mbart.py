import os

import mbart
import pytest
from unittest.mock import call
import pandas as pd


t = (
    mbart.mbart_Translator()
)  # I have created an instance of mbart_Translator here to use it in all tests


def test_mbart_Translator():
    # all testing functions have to start with "test_"
    # This function is for checking the attributes of mbart_Translator class if all the attributes are initialized correctly on creating instances of the class
    assert t.module_name == "facebook/mbart-large-50-many-to-many-mmt"
    assert t.source_language == "ja_XX"
    assert t.target_language == "en_XX"
    assert t.tokenizer.src_lang == t.source_language


class TestTranslate(object):
    # This class is created for testing translate() method of mbart_Translator class which is supposed to translate a single string
    # While writing test classes you always put "object" as an argument
    # all test classes have to start with "Test"

    def test_blank_string(self):
        # testing when input string is blank
        test_argument = ""
        expected = "In fact, it's not as if we're going to be able to get rid of it."  # this is the translation output from mbart model, it is very bad translation but this is what we got
        actual = t.translate(text=test_argument)
        message = "Wrong blank translation, expected is {0}, but given {1}".format(
            expected, actual
        )  # is possible to leave a message when the test fails
        assert isinstance(
            actual, str
        )  # isinstance checks whether output is in given data type
        assert (
            actual == expected
        ), message  # checks if the actual output equals to given one

    def test_none_input(self):
        # This is for testing whether the function gives ValueError if a None input is given
        test_argument = None
        with pytest.raises(ValueError):
            t.translate(text=test_argument)

    def test_normal_sentence(self):
        # This is a normal case, mbart translates "adsa" to "adsa"
        test_argument = "adsa"
        expected = "adsa"
        actual = t.translate(text=test_argument)
        message = "Wrong translation, expected is {0}, but given {1}".format(
            expected, actual
        )
        assert isinstance(actual, str)
        assert actual == expected, message

    def test_type_error(self):
        # Testing whether the function gives ValueError when a int input is given which is non-string
        test_argument = 9
        with pytest.raises(ValueError):
            t.translate(text=test_argument)

    def test_normal_sentence_2(self):
        # testing with a japanese sentence
        test_argument = "加藤産業㈱阪神支店ﾘﾍﾞｰﾄ3月"
        expected = "Kato Industries (株) Hanshin Branch Rehearsal March"
        actual = t.translate(text=test_argument)
        message = "Wrong translation, expected is {0}, but given {1}".format(
            expected, actual
        )
        assert isinstance(actual, str)
        assert actual == expected, message


@pytest.fixture  # fixture decorator should be used when a file creation in a path needs to be tested
def saving_data_file():
    # this function will be used below and has path data
    file_data_path = "trial.csv"
    yield file_data_path
    os.remove(file_data_path)


def test_on_saving_data(saving_data_file):
    # this function is for testing save_csv method of mbart_Translator class which saves a pandas dataframe to csv
    # by saving a file to csv, you modify the environment on a given path, this type of environment modifications can be tested with pytest
    # by using the fixture on the top, the file will be deleted after testing, so no memory issues!
    file_path = saving_data_file
    mydata = {
        "text": ["adsa", "加藤産業㈱阪神支店ﾘﾍﾞｰﾄ3月"],
        "translation": ["adsa", "Kato Industries (株) Hanshin Branch Rehearsal March",],
    }
    mydata = pd.DataFrame(data=mydata)  # example dataframe to be tested
    t.save_csv(dataframe=mydata)
    saved_data = pd.read_csv(file_path)
    first_text = saved_data["text"].iloc[0]
    second_text = saved_data["text"].iloc[1]
    first_translation = saved_data["translation"].iloc[0]
    second_translation = saved_data["translation"].iloc[1]
    # we check if the saved file contains the correct information
    assert first_text == "adsa"
    assert second_text == "加藤産業㈱阪神支店ﾘﾍﾞｰﾄ3月"
    assert first_translation == "adsa"
    assert second_translation == "Kato Industries (株) Hanshin Branch Rehearsal March"


class TestColumnTranslate(object):
    # this class is for testing column_translate() method in mbart_Translation class which takes a pandasdataframe column ,translates it, it gives a list of sentences as an output
    # the issue here is this method uses the method translate() from same class, thus have a dependency
    # you need to apply mocking to overcome the dependency, in this example we will mock the translate() function
    def translate_bug_free(self, text: str):
        # need to have same arguments with original function
        # this is bug free version of translation() to mock it, you define a dictionary with example inputs as keys and corresponding outputs as values
        return_values = {
            "adsa": "adsa",
            "加藤産業㈱阪神支店ﾘﾍﾞｰﾄ3月": "Kato Industries (株) Hanshin Branch Rehearsal March",
        }
        return return_values[text]

    def test_normal_column_translation(self, mocker):
        # this function is for testing column_translate()method by mocking translate() which is a dependency
        mydata = {
            "text": ["adsa", "加藤産業㈱阪神支店ﾘﾍﾞｰﾄ3月"],
            "translation": [
                "adsa",
                "Kato Industries (株) Hanshin Branch Rehearsal March",
            ],
        }
        mydata = pd.DataFrame(
            data=mydata
        )  # example  dataframe with inputs and correct translations
        translate_mock = mocker.patch(
            "mbart.mbart_Translator.translate", side_effect=self.translate_bug_free
        )  # this mocks given function with bugfree one
        translated_list = t.column_translate(
            dataframe=mydata, colum_name="text"
        )  # call the method by example dataframe
        assert translate_mock.call_args_list == [
            call(text="adsa"),
            call(text="加藤産業㈱阪神支店ﾘﾍﾞｰﾄ3月"),
        ]  # checking if the bug_free version is called with correct arguments

        assert isinstance(translated_list, list)
        assert (
            translated_list[0] == mydata["translation"][0]
        )  # checking whether the method works correctly free of dependencies
        assert translated_list[1] == mydata["translation"][1]
