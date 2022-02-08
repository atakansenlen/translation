from typing import List
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd


class Translator:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-ja-en") -> None:
        self.model_name = model_name
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name)

    def translate(self, text: str) -> str:
        src_text = [text]
        translated = self.model.generate(
            **self.tokenizer(src_text, return_tensors="pt", padding=True)
        )
        tgt_text = [
            self.tokenizer.decode(t, skip_special_tokens=True) for t in translated
        ]
        return str(tgt_text[0])

    def column_translate(self, dataframe: pd.DataFrame, colum_name: str) -> List[str]:
        return [self.translate(text=x) for x in dataframe[colum_name]]

    def bulk_translate(self, dataframe: pd.DataFrame, colum_name: str) -> List[str]:
        mybulk = dataframe[colum_name].tolist()
        translated = self.model.generate(
            **self.tokenizer(mybulk, return_tensors="pt", padding=True)
        )
        tgt_text = [
            self.tokenizer.decode(t, skip_special_tokens=True) for t in translated
        ]
        return tgt_text

