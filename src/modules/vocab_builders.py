from sklearn.feature_extraction.text import CountVectorizer

from src.utils.mapper import ConfigMapper


@ConfigMapper.map("vocab_builders", "word_frequency_vocab_builder")
class WordFrequencyVocabBuilder:
    def __init__(self, config):
        self.min_freq = config.min_freq

    def __call__(self, df, text_col_name, vocab_save_path):
        print("\nBuilding Vocabulary Based on Word Frequency...")
        vectorizer = CountVectorizer(
            min_df=self.min_freq,
            tokenizer=lambda x: x.split(" "),
        )
        vectorizer.fit(df[text_col_name])
        with open(vocab_save_path, "w") as fout:
            for word in vectorizer.get_feature_names():
                fout.write(f"{word}\n")
