import logging
import re
import sys

import nltk
from nltk.corpus import stopwords
from nltk.stem import (
    LancasterStemmer,
    PorterStemmer,
    RSLPStemmer,
    SnowballStemmer,
    WordNetLemmatizer,
)
from nltk.tokenize import RegexpTokenizer

from src.utils.file_loaders import load_json
from src.utils.text_loggers import get_logger

logger = get_logger(__name__)

nltk.download("omw-1.4")
nltk.download("rslp")
nltk.download("stopwords")
nltk.download("wordnet")

# Clinical Note preprocessing
AVAILABLE_STEMMERS_LEMMATIZERS = {
    "nltk.LancasterStemmer": LancasterStemmer,
    "nltk.PorterStemmer": PorterStemmer,
    "nltk.RSLPStemmer": RSLPStemmer,
    "nltk.SnowballStemmer": SnowballStemmer,
    "nltk.WordNetLemmatizer": WordNetLemmatizer,
}


class ClinicalNotePreprocessor:
    def __init__(self, config):
        self._config = config
        logger.debug(
            "Initialising Clinical Note Processor with the following "
            "config: {}".format(config.as_dict())
        )
        self.punct_tokenizer = RegexpTokenizer(r"\w+")

        if config.remove_stopwords.perform:
            stopwords_file_path = (
                config.remove_stopwords.params.stopwords_file_path
            )
            if stopwords_file_path:
                self.stopword_list = set(load_json(stopwords_file_path))
            else:
                self.stopword_list = set(stopwords.words("english"))
            if config.remove_stopwords.params.remove_common_medical_terms:
                # add a few common terms used in medicine
                self.stopword_list.update(
                    {
                        "admission",
                        "birth",
                        "date",
                        "discharge",
                        "service",
                        "sex",
                        "patient",
                        "name",
                        "history",
                        "hospital",
                        "last",
                        "first",
                        "course",
                        "past",
                        "day",
                        "one",
                        "family",
                        "chief",
                        "complaint",
                    }
                )

        if config.stem_or_lemmatize.perform:
            if (
                config.stem_or_lemmatize.params.stemmer_name
                == "nltk.SnowballStemmer"
            ):
                self.stemmer = AVAILABLE_STEMMERS_LEMMATIZERS[
                    config.stem_or_lemmatize.params.stemmer_name
                ]("english")
            else:
                self.stemmer = AVAILABLE_STEMMERS_LEMMATIZERS[
                    config.stem_or_lemmatize.params.stemmer_name
                ]()

    def __call__(self, text):
        # Remove extra spaces from text
        text = re.sub(" +", " ", text).strip()
        if self._config.to_lower.perform:
            text = text.lower()

        if self._config.remove_punctuation.perform:
            tokens = self.remove_punctuation(text)
        else:
            tokens = text.split(" ")

        if self._config.remove_numeric.perform:
            if self._config.remove_numeric.replace_numerics_with_letter:
                tokens = self.remove_numeric(
                    tokens,
                    self._config.remove_numeric.replace_numerics_with_letter,
                )
            else:
                tokens = self.remove_numeric(tokens)

        if self._config.remove_stopwords.perform:
            tokens = self.remove_stopwords(tokens)

        if self._config.stem_or_lemmatize.perform:
            tokens = self.stem_or_lemmatize(tokens)

        if self._config.truncate.perform:
            if len(tokens) > self._config.truncate.params.max_length:
                tokens = tokens[: self._config.truncate.params.max_length]

        return " ".join(tokens)

    def to_lower_case(self, text):
        return text.lower()

    def remove_punctuation(self, text):
        tokens = self.punct_tokenizer.tokenize(text)
        return tokens

    def remove_numeric(self, tokens, replace_numerics_with_letter=None):
        if replace_numerics_with_letter is not None:
            tokens = [
                re.sub("\\d", replace_numerics_with_letter, t)
                for t in tokens
                if not t.isnumeric()
            ]
        else:
            tokens = [t for t in tokens if not t.isnumeric()]
        return tokens

    def replace_numerics_with_letter(self, tokens, letter):
        return [letter if t.isnumeric() else t for t in tokens]

    def remove_stopwords(self, tokens):
        return [t for t in tokens if t not in self.stopword_list]

    def stem_or_lemmatize(self, tokens):
        if (
            self._config.stem_or_lemmatize.params.stemmer_name
            == "nltk.WordNetLemmatizer"
        ):
            return [self.stemmer.lemmatize(t) for t in tokens]
        else:
            return [self.stemmer.stem(t) for t in tokens]


class CodeProcessor:
    def __init__(self, config):
        self._config = config
        logger.debug(
            "Initialising Code Processor with the following config: {}".format(
                config.as_dict()
            )
        )

    def __call__(self, icd_code, is_diagnosis_code):
        if self._config.add_period_in_correct_pos.perform:
            icd_code = self.reformat_icd_code(icd_code, is_diagnosis_code)
        return icd_code

    @staticmethod
    def reformat_icd_code(icd_code, is_diagnosis_code):
        code = "".join(icd_code.split("."))
        if is_diagnosis_code:
            if code.startswith("E"):
                if len(code) > 4:
                    code = code[:4] + "." + code[4:]
            else:
                if len(code) > 3:
                    code = code[:3] + "." + code[3:]
        else:
            if len(code) > 2:
                code = code[:2] + "." + code[2:]
        return code
