import csv
import os
import time

import pandas as pd
from tqdm.auto import tqdm

from src.modules.dataset_splitters import *
from src.modules.embeddings import *
from src.modules.preprocessors import ClinicalNotePreprocessor, CodeProcessor
from src.modules.tokenizers import *
from src.utils.code_based_filtering import TopKCodes
from src.utils.file_loaders import load_csv_as_df, save_df, save_json
from src.utils.mapper import ConfigMapper
from src.utils.text_loggers import get_logger

logger = get_logger(__name__)
tqdm.pandas()


@ConfigMapper.map("preprocessing_pipelines", "mimic_iii_preprocessing_pipeline")
class MimiciiiPreprocessingPipeline:
    def __init__(self, config):
        self.config = config
        self.MIMIC_DIR = config.paths.mimic_dir
        self.SAVE_DIR = config.paths.save_dir
        self.cols = config.dataset_metadata.column_names
        self.clinical_note_config = config.clinical_note_preprocessing
        self.code_config = config.code_preprocessing

        self.train_json_name = os.path.join(
            self.SAVE_DIR, config.paths.train_json_name
        )
        self.val_json_name = os.path.join(
            self.SAVE_DIR, config.paths.val_json_name
        )
        self.test_json_name = os.path.join(
            self.SAVE_DIR, config.paths.test_json_name
        )

        if not os.path.exists(self.MIMIC_DIR):
            os.makedirs(self.MIMIC_DIR)

        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)

        self.clinical_note_preprocessor = ClinicalNotePreprocessor(
            self.clinical_note_config
        )
        self.code_preprocessor = CodeProcessor(self.code_config)

        self.top_k_codes = TopKCodes(
            k=self.code_config.top_k,
            labels_save_path=os.path.join(
                self.SAVE_DIR, config.paths.label_json_name
            ),
            labels_freq_save_path=os.path.join(
                self.SAVE_DIR, config.paths.label_freq_json_name
            )
            if config.paths.label_freq_json_name is not None
            else None,
        )
        self.split_data = ConfigMapper.get_object(
            "dataset_splitters", config.dataset_splitting_method.name
        )(config.dataset_splitting_method.params)

        self.tokenizer = ConfigMapper.get_object(
            "tokenizers", config.tokenizer.name
        )(config.tokenizer.params)

        self.embedder = ConfigMapper.get_object(
            "embeddings", config.embedding.name
        )(config.embedding.params)

        self.code_csv_dtypes = {
            self.cols.hadm_id: "string",
            self.cols.icd9_code: "string",
        }
        self.noteevents_csv_dtypes = {
            self.cols.hadm_id: "string",
            self.cols.text: "string",
        }

    def extract_df_based_on_code_type(self):
        code_type = self.code_config.code_type

        diagnosis_code_csv_path = os.path.join(
            self.MIMIC_DIR, self.config.paths.diagnosis_code_csv_name
        )
        procedure_code_csv_path = os.path.join(
            self.MIMIC_DIR, self.config.paths.procedure_code_csv_name
        )
        logger.info(
            "Loading code CSV files: {}, {}".format(
                diagnosis_code_csv_path, procedure_code_csv_path
            )
        )
        assert code_type in [
            "diagnosis",
            "procedure",
            "both",
        ], 'code_type should be one of ["diagnosis", "procedure", "both"]'

        if not self.config.incorrect_code_loading:
            diagnosis_code_df = load_csv_as_df(
                diagnosis_code_csv_path, dtype=self.code_csv_dtypes
            )
            procedure_code_df = load_csv_as_df(
                procedure_code_csv_path, dtype=self.code_csv_dtypes
            )
        else:
            # CAML's notebook does not specify dtype
            ccol = self.cols.icd9_code
            diagnosis_code_df = load_csv_as_df(diagnosis_code_csv_path)
            procedure_code_df = load_csv_as_df(procedure_code_csv_path)
            diagnosis_code_df[ccol] = diagnosis_code_df[ccol].astype("string")
            procedure_code_df[ccol] = procedure_code_df[ccol].astype("string")
        logger.info(
            "Preprocessing code CSV files: {}, {}".format(
                diagnosis_code_csv_path, procedure_code_csv_path
            )
        )
        diagnosis_code_df[self.cols.icd9_code] = diagnosis_code_df[
            self.cols.icd9_code
        ].apply(
            lambda x: str(self.code_preprocessor(str(x), True)),
        )
        procedure_code_df[self.cols.icd9_code] = procedure_code_df[
            self.cols.icd9_code
        ].apply(
            lambda x: str(self.code_preprocessor(str(x), False)),
        )

        if code_type == "diagnosis":
            code_df = diagnosis_code_df
        elif code_type == "procedure":
            code_df = procedure_code_df
        else:
            code_df = pd.concat([diagnosis_code_df, procedure_code_df])

        # Delete unnecessary columns. (When we are not reprocuding CAML's ver)
        if not self.config.incorrect_code_loading:
            code_df = code_df[
                [
                    self.cols.hadm_id,
                    self.cols.icd9_code,
                ]
            ]
        return code_df

    def filter_icd_codes_based_on_clinical_notes(self, code_df, noteevents_df):
        logger.info(
            "Removing rows from code dataframe whose ICD-9 codes are not "
            "present in clinical notes"
        )

        hadm_ids = set(noteevents_df[self.cols.hadm_id])
        if not self.config.incorrect_code_loading:
            code_df = code_df[code_df[self.cols.hadm_id].isin(hadm_ids)]
        else:
            # Reproduce CAML notebook's behavior
            temp_fpath = f"temp_{time.time()}.csv"
            scol = self.cols.subject_id
            hcol = self.cols.hadm_id
            ccol = self.cols.icd9_code
            with open(temp_fpath, "w") as fd:
                w = csv.writer(fd)
                w.writerow([scol, hcol, ccol, "ADMITTIME", "DISCHTIME"])
                for _, row in code_df.iterrows():
                    if str(row.HADM_ID) in hadm_ids:
                        w.writerow(
                            [row.SUBJECT_ID, row.HADM_ID, row.ICD9_CODE, "", ""]
                        )
            code_df = pd.read_csv(temp_fpath, index_col=None)
            code_df = code_df.sort_values([scol, hcol])
            code_df.to_csv(temp_fpath, index=False)
            code_df = pd.read_csv(temp_fpath, index_col=None)
            code_df[hcol] = code_df[hcol].astype("string")
            code_df[ccol] = code_df[ccol].astype("string")
            code_df = code_df[[self.cols.hadm_id, self.cols.icd9_code]]
            os.remove(temp_fpath)

        return code_df

    def preprocess_clinical_note(self, clinical_note):
        clinical_note = self.clinical_note_preprocessor(clinical_note)
        return clinical_note

    def load_clinical_notes(self):
        notes_file_path = os.path.join(
            self.MIMIC_DIR, self.config.paths.noteevents_csv_name
        )

        logger.info("Loading noteevents CSV file: {}".format(notes_file_path))

        noteevents_df = load_csv_as_df(
            notes_file_path, dtype=self.noteevents_csv_dtypes
        )
        # To-do: Add other categories later, based on args provided by the user
        noteevents_df = noteevents_df[
            noteevents_df[self.cols.category] == "Discharge summary"
        ]

        # Delete unnecessary columns
        noteevents_df = noteevents_df[
            [
                self.cols.hadm_id,
                self.cols.text,
            ]
        ]
        return noteevents_df

    def combine_code_and_notes(self, code_df, noteevents_df):
        logger.info("Combining code and notes dataframes")
        noteevents_grouped = noteevents_df.groupby(self.cols.hadm_id)[
            self.cols.text
        ].apply(lambda texts: " ".join(texts))
        noteevents_df = pd.DataFrame(noteevents_grouped)
        noteevents_df.reset_index(inplace=True)

        # Preprocess clinical notes
        logger.info("Preprocessing clinical notes")
        noteevents_df[self.cols.text] = noteevents_df[
            self.cols.text
        ].progress_map(self.preprocess_clinical_note)

        if not self.config.incorrect_code_loading:
            codes_grouped = code_df.groupby(self.cols.hadm_id)[
                self.cols.icd9_code
            ].apply(
                lambda codes: ";".join(map(str, list(dict.fromkeys(codes))))
            )
        else:
            # CAML's notebook counts duplicated ICD codes separately
            codes_grouped = code_df.groupby(self.cols.hadm_id)[
                self.cols.icd9_code
            ].apply(lambda codes: ";".join(codes))
        code_df = pd.DataFrame(codes_grouped)
        code_df.reset_index(inplace=True)

        combined_df = pd.merge(noteevents_df, code_df, on=self.cols.hadm_id)
        combined_df.sort_values(
            [self.cols.hadm_id], inplace=True, ignore_index=True
        )
        combined_df.rename(
            columns={self.cols.icd9_code: self.cols.labels}, inplace=True
        )

        return combined_df

    def preprocess(self):
        code_df = self.extract_df_based_on_code_type()
        noteevents_df = self.load_clinical_notes()
        code_df = self.filter_icd_codes_based_on_clinical_notes(
            code_df, noteevents_df
        )
        combined_df = self.combine_code_and_notes(code_df, noteevents_df)
        combined_df = self.top_k_codes(self.cols.labels, combined_df)

        logger.info("Splitting data into train-test-val")
        train_df, val_df, test_df = self.split_data(
            combined_df, self.cols.hadm_id
        )

        # convert dataset to dictionary
        train_df = train_df.to_dict(orient="list")
        val_df = val_df.to_dict(orient="list")
        test_df = test_df.to_dict(orient="list")

        # tokenize the data
        logger.info("Tokenizing text data")
        train_df[self.cols.text] = self.tokenizer.tokenize_list(
            train_df[self.cols.text]
        )
        val_df[self.cols.text] = self.tokenizer.tokenize_list(
            val_df[self.cols.text]
        )
        test_df[self.cols.text] = self.tokenizer.tokenize_list(
            test_df[self.cols.text]
        )

        save_json(
            train_df,
            os.path.join(
                self.config.paths.save_dir, self.config.paths.train_json_name
            ),
        )
        save_json(
            val_df,
            os.path.join(
                self.config.paths.save_dir, self.config.paths.val_json_name
            ),
        )
        save_json(
            test_df,
            os.path.join(
                self.config.paths.save_dir, self.config.paths.test_json_name
            ),
        )

        # train embedding model
        logger.info("Training embedding model")
        if self.config.train_embed_with_all_split:
            all_text = (
                train_df[self.cols.text]
                + val_df[self.cols.text]
                + test_df[self.cols.text]
            )
        else:
            all_text = train_df[self.cols.text]
        self.embedder.train(all_text)
