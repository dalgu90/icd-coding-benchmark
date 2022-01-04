from src.modules.tokenizers import *
from src.modules.embeddings import *
from src.utils.mapper import configmapper


class Preprocessor:
    def preprocess(self):
        pass


@configmapper.map("preprocessors", "glove")
class GlovePreprocessor(Preprocessor):
    """GlovePreprocessor."""

    def __init__(self, config):
        """
        Args:
            config (src.utils.module.Config): configuration for preprocessor
        """
        super(GlovePreprocessor, self).__init__()
        self.config = config
        self.tokenizer = configmapper.get_object(
            "tokenizers", self.config.main.preprocessor.tokenizer.name
        )(**self.config.main.preprocessor.tokenizer.init_params.as_dict())
        self.tokenizer_params = (
            self.config.main.preprocessor.tokenizer.init_vector_params.as_dict()
        )

        self.tokenizer.initialize_vectors(**self.tokenizer_params)
        self.embeddings = configmapper.get_object(
            "embeddings", self.config.main.preprocessor.embedding.name
        )(
            self.tokenizer.text_field.vocab.vectors,
            self.tokenizer.text_field.vocab.stoi[self.tokenizer.text_field.pad_token],
        )

    def preprocess(self, model_config, data_config):
        train_dataset = configmapper.get_object("datasets", data_config.main.name)(
            data_config.train, self.tokenizer
        )
        val_dataset = configmapper.get_object("datasets", data_config.main.name)(
            data_config.val, self.tokenizer
        )
        model = configmapper.get_object("models", model_config.name)(
            self.embeddings, **model_config.params.as_dict()
        )

        return model, train_dataset, val_dataset


@configmapper.map("preprocessors", "clozePreprocessor")
class ClozePreprocessor(Preprocessor):
    """GlovePreprocessor."""

    def __init__(self, config):
        """
        Args:
            config (src.utils.module.Config): configuration for preprocessor
        """
        super(ClozePreprocessor, self).__init__()
        self.config = config
        self.tokenizer = configmapper.get_object(
            "tokenizers", self.config.main.preprocessor.tokenizer.name
        ).from_pretrained(
            **self.config.main.preprocessor.tokenizer.init_params.as_dict()
        )

    def preprocess(self, model_config, data_config):
        train_dataset = configmapper.get_object("datasets", data_config.main.name)(
            data_config.train, self.tokenizer
        )
        val_dataset = configmapper.get_object("datasets", data_config.main.name)(
            data_config.val, self.tokenizer
        )
        model = configmapper.get_object("models", model_config.name).from_pretrained(
            **model_config.params.as_dict()
        )

        return model, train_dataset, val_dataset


@configmapper.map("preprocessors", "transformersConcretenessPreprocessor")
class TransformersConcretenessPreprocessor(Preprocessor):
    """BertConcretenessPreprocessor."""

    def __init__(self, config):
        """
        Args:
            config (src.utils.module.Config): configuration for preprocessor
        """
        super(TransformersConcretenessPreprocessor, self).__init__()
        self.config = config
        self.tokenizer = configmapper.get_object(
            "tokenizers", self.config.main.preprocessor.tokenizer.name
        ).from_pretrained(
            **self.config.main.preprocessor.tokenizer.init_params.as_dict()
        )

    def preprocess(self, model_config, data_config):

        train_dataset = configmapper.get_object("datasets", data_config.main.name)(
            data_config.train, self.tokenizer
        )
        val_dataset = configmapper.get_object("datasets", data_config.main.name)(
            data_config.val, self.tokenizer
        )

        model = configmapper.get_object("models", model_config.name)(
            **model_config.params.as_dict()
        )

        return model, train_dataset, val_dataset
