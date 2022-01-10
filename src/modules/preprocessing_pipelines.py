import os

import pandas as pd
from tqdm import tqdm

from src.modules.dataset_splitters import *
from src.modules.preprocessors import (
    ReformatICDCode,
    RemoveNumericOnlyTokens,
    ToLowerCase,
)
from src.utils.code_based_filtering import TopKCodes
from src.utils.file_loaders import load_csv_as_df
from src.utils.mapper import ConfigMapper

tqdm.pandas()


@ConfigMapper.map("preprocessing_pipelines", "mimic_iii_preprocessing_pipeline")
class MimiciiiPreprocessingPipeline:
    def __init__(self, config):
        self.config = config
        self.MIMIC_DIR = config.dirs.mimic_dir
        self.cols = config.dataset_metadata.column_names
        self.clinical_note_config = config.clinical_note_preprocessing
        self.code_config = config.code_preprocessing

        self.top_k_codes = TopKCodes(self.code_config.top_k)
        self.split_data = ConfigMapper.get_object(
            "dataset_splitters", config.dataset_splitting_method.name
        )(config.dataset_splitting_method.params)

    def extract_df_based_on_code_type(self):
        code_type = self.code_config.code_type
        add_period_in_correct_pos = self.code_config.add_period_in_correct_pos

        diagnosis_code_csv_path = os.path.join(
            self.MIMIC_DIR, self.config.dirs.diagnosis_code_csv_name
        )
        procedure_code_csv_path = os.path.join(
            self.MIMIC_DIR, self.config.dirs.procedure_code_csv_name
        )
        assert code_type in [
            "diagnosis",
            "procedure",
            "both",
        ], 'code_type should be one of ["diagnosis", "procedure", "both"]'

        diagnosis_code_df = load_csv_as_df(diagnosis_code_csv_path)
        procedure_code_df = load_csv_as_df(procedure_code_csv_path)

        if add_period_in_correct_pos:
            reformat_icd_code = ReformatICDCode()
            diagnosis_code_df[self.cols.icd9_code] = diagnosis_code_df.apply(
                lambda row: str(reformat_icd_code(str(row[4]), True)), axis=1
            )
            procedure_code_df[self.cols.icd9_code] = procedure_code_df.apply(
                lambda row: str(reformat_icd_code(str(row[4]), False)), axis=1
            )

        if code_type == "diagnosis":
            code_df = diagnosis_code_df
        elif code_type == "procedure":
            code_df = procedure_code_df
        else:
            code_df = pd.concat([diagnosis_code_df, procedure_code_df])
        return code_df

    def filter_icd_codes_based_on_clinical_notes(self, code_df, noteevents_df):
        hadm_ids = set(noteevents_df[self.cols.hadm_id])
        code_df = code_df[code_df[self.cols.hadm_id].isin(hadm_ids)]
        return code_df

    def preprocess_clinical_note(self, clinical_note):
        if self.clinical_note_config.to_lower.perform:
            to_lower_case = ToLowerCase()
            clinical_note = to_lower_case(clinical_note)

        if self.clinical_note_config.remove_punc_numeric_tokens.perform:
            remove_numeric_only_tokens = RemoveNumericOnlyTokens()
            clinical_note = remove_numeric_only_tokens(clinical_note)

        return clinical_note

    def preprocess_clinical_notes(self):
        print("\nProcessing Clinical Notes...")
        notes_file_path = os.path.join(
            self.MIMIC_DIR, self.config.dirs.noteevents_csv_name
        )

        noteevents_df = load_csv_as_df(notes_file_path)
        # To-do: Add other categories later, based on args provided by the user
        noteevents_df = noteevents_df[
            noteevents_df[self.cols.category] == "Discharge summary"
        ]
        # Preprocess clinical notes
        noteevents_df[self.cols.text] = noteevents_df[
            self.cols.text
        ].progress_map(self.preprocess_clinical_note)
        # Delete unnecessary columns
        noteevents_df = noteevents_df[
            [
                self.cols.subject_id,
                self.cols.hadm_id,
                self.cols.charttime,
                self.cols.text,
            ]
        ]
        return noteevents_df

    def combine_code_and_notes(self, code_df, noteevents_df):
        print("\nForming Text-Label Dataframe...")
        # Sort by SUBJECT_ID and HADM_ID
        noteevents_df = noteevents_df.sort_values(
            [self.cols.subject_id, self.cols.hadm_id]
        )
        code_df = code_df.sort_values([self.cols.subject_id, self.cols.hadm_id])

        subj_id_hadm_id_list = list(
            set(
                zip(code_df[self.cols.subject_id], code_df[self.cols.hadm_id])
            ).intersection(
                set(
                    zip(
                        noteevents_df[self.cols.subject_id],
                        code_df[self.cols.hadm_id],
                    )
                )
            )
        )
        final_df = pd.DataFrame(
            columns=[
                self.cols.subject_id,
                self.cols.hadm_id,
                self.cols.text,
                "label",
            ]
        )
        for subj_id, hadm_id in tqdm(subj_id_hadm_id_list):
            code_df_rows = code_df[
                (code_df[self.cols.subject_id] == subj_id)
                & (code_df[self.cols.hadm_id] == hadm_id)
            ]
            noteevents_df_rows = noteevents_df[
                (noteevents_df[self.cols.subject_id] == subj_id)
                & (noteevents_df[self.cols.hadm_id] == hadm_id)
            ]

            codes = []
            notes = []
            for _, row in code_df_rows.iterrows():
                codes.append(row[self.cols.icd9_code])
            for _, row in noteevents_df_rows.iterrows():
                notes.append(row[self.cols.text])

            final_df = final_df.append(
                {
                    self.cols.subject_id: subj_id,
                    self.cols.hadm_id: hadm_id,
                    self.cols.text: " ".join(notes).strip(),
                    "label": ";".join(codes),
                },
                ignore_index=True,
            )

        return final_df

    def preprocess(self):
        code_df = self.extract_df_based_on_code_type()
        noteevents_df = self.preprocess_clinical_notes()
        code_df = self.filter_icd_codes_based_on_clinical_notes(
            code_df, noteevents_df
        )
        combined_df = self.combine_code_and_notes(code_df, noteevents_df)
        combined_df = self.top_k_codes("label", combined_df)
        train_df, val_df, test_df = self.split_data(
            combined_df, self.cols.hadm_id
        )
        return (train_df, val_df, test_df)
