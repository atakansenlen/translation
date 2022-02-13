import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from typing import List


class mbart_Translator:
    def __init__(
        self,
        module_name: str = "facebook/mbart-large-50-many-to-many-mmt",
        source_language: str = "ja_XX",
        target_language: str = "en_XX",
    ) -> None:
        self.module_name = module_name
        self.source_language = source_language
        self.target_language = target_language
        self.model = MBartForConditionalGeneration.from_pretrained(self.module_name)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(self.module_name)
        self.tokenizer.src_lang = self.source_language

    def translate(self, text: str) -> str:
        """translates a single string from source language to target language

        Args:
            text (str): Source text which is desired to translate

        Returns:
            str: Translation of a single string from source language to target language
        """
        encoded_hi = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(
            **encoded_hi,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.target_language]
        )
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[
            0
        ]

    def column_translate(self, dataframe: pd.DataFrame, colum_name: str) -> List[str]:
        """Translates a given column of a pandas dataframe from source language to a list in target language

        Args:
            dataframe (pd.DataFrame): Given data frame which contains the related data
            colum_name (str): Given column which contains strings for the translation

        Returns:
            List[str]: list of strings which are translated from source language to target language
        """
        return [self.translate(text=x) for x in dataframe[colum_name]]

    def save_csv(self, dataframe: pd.DataFrame, index: bool = False) -> None:
        """Saves a given pandas dataframe to a csv file by using to_csv method of pd.DataFrame

        Args:
            dataframe (pd.DataFrame): given pandas dataframe
            index (bool, optional): indicates if indexes are desired to be kept in csv. Defaults to False.

        Returns:
            [type]: no return
        """
        dataframe.to_csv("trial.csv", index=index)
        return None

