import os

import pandas as pd
from tqdm.auto import tqdm

from src.modules.dataset_splitters import *
from src.modules.preprocessors import (
    ReformatICDCode,
    RemoveNumericOnlyTokens,
    ToLowerCase,
)
from src.utils.code_based_filtering import TopKCodes
from src.utils.file_loaders import load_csv_as_df, save_df
from src.utils.mapper import ConfigMapper

tqdm.pandas()


@ConfigMapper.map("preprocessing_pipelines", "mimic_iii_preprocessing_pipeline")
class MimiciiiPreprocessingPipeline:
    def __init__(self, config):
        self.config = config
        self.MIMIC_DIR = config.paths.mimic_dir
        self.cols = config.dataset_metadata.column_names
        self.clinical_note_config = config.clinical_note_preprocessing
        self.code_config = config.code_preprocessing

        self.train_csv_name = os.path.join(
            self.MIMIC_DIR, config.paths.train_csv_name
        )
        self.val_csv_name = os.path.join(
            self.MIMIC_DIR, config.paths.val_csv_name
        )
        self.test_csv_name = os.path.join(
            self.MIMIC_DIR, config.paths.test_csv_name
        )

        self.top_k_codes = TopKCodes(
            self.code_config.top_k,
            os.path.join(self.MIMIC_DIR, config.paths.labels_json_name),
        )
        self.split_data = ConfigMapper.get_object(
            "dataset_splitters", config.dataset_splitting_method.name
        )(config.dataset_splitting_method.params)

        self.code_csv_dtypes = {
            self.cols.subject_id: "string",
            self.cols.hadm_id: "string",
            self.cols.icd9_code: "string",
        }
        self.noteevents_csv_dtypes = {
            self.cols.subject_id: "string",
            self.cols.hadm_id: "string",
            self.cols.text: "string",
        }

    def extract_df_based_on_code_type(self):
        code_type = self.code_config.code_type
        add_period_in_correct_pos = self.code_config.add_period_in_correct_pos

        diagnosis_code_csv_path = os.path.join(
            self.MIMIC_DIR, self.config.paths.diagnosis_code_csv_name
        )
        procedure_code_csv_path = os.path.join(
            self.MIMIC_DIR, self.config.paths.procedure_code_csv_name
        )
        assert code_type in [
            "diagnosis",
            "procedure",
            "both",
        ], 'code_type should be one of ["diagnosis", "procedure", "both"]'

        diagnosis_code_df = load_csv_as_df(
            diagnosis_code_csv_path, dtype=self.code_csv_dtypes
        )
        procedure_code_df = load_csv_as_df(
            procedure_code_csv_path, dtype=self.code_csv_dtypes
        )

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
            self.MIMIC_DIR, self.config.paths.noteevents_csv_name
        )

        noteevents_df = load_csv_as_df(
            notes_file_path, dtype=self.noteevents_csv_dtypes
        )
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
            [self.cols.subject_id, self.cols.hadm_id], ignore_index=True
        )
        code_df = code_df.sort_values(
            [self.cols.subject_id, self.cols.hadm_id], ignore_index=True
        )

        final_df = pd.DataFrame(
            columns=[
                self.cols.subject_id,
                self.cols.hadm_id,
                self.cols.text,
                "label",
            ]
        )

        code_df_iter = code_df.iterrows()
        code_df_idx, code_df_sample = next(code_df_iter)
        code_df_size = len(code_df)
        code_df_flag = True

        noteevents_df_iter = noteevents_df.iterrows()
        noteevents_df_idx, noteevents_df_sample = next(noteevents_df_iter)
        noteevents_df_size = len(noteevents_df)
        noteevents_df_flag = True

        curr_subj_id = code_df_sample[self.cols.subject_id]
        curr_hadm_id = code_df_sample[self.cols.hadm_id]

        while True:
            codes = []
            notes = []
            while True:
                # If the current row has its hadm_id = curr_hadm_id, then append
                # to list. Otherwise, break.
                if code_df_sample[self.cols.hadm_id] != curr_hadm_id:
                    break
                else:
                    codes.append(code_df_sample[self.cols.icd9_code])

                # Iterate to the next row.
                if code_df_idx < code_df_size - 1:
                    code_df_idx, code_df_sample = next(code_df_iter)
                else:
                    code_df_flag = False
                    break

            while True:
                # If the current row has its hadm_id = curr_hadm_id, then append
                # to list. Otherwise, break.
                if noteevents_df_sample[self.cols.hadm_id] != curr_hadm_id:
                    break
                else:
                    notes.append(noteevents_df_sample[self.cols.text])

                # Iterate to the next row.
                if noteevents_df_idx < noteevents_df_size - 1:
                    noteevents_df_idx, noteevents_df_sample = next(
                        noteevents_df_iter
                    )
                else:
                    noteevents_df_flag = False
                    break

            if len(codes) > 0 and len(notes) > 0:
                final_df = final_df.append(
                    {
                        self.cols.subject_id: curr_subj_id,
                        self.cols.hadm_id: curr_hadm_id,
                        self.cols.text: " ".join(notes).strip(),
                        "label": ";".join(codes),
                    },
                    ignore_index=True,
                )
            curr_hadm_id = code_df_sample[self.cols.hadm_id]
            curr_subj_id = code_df_sample[self.cols.subject_id]
            if not (code_df_flag and noteevents_df_flag):
                break

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
        save_df(train_df, self.train_csv_name)
        save_df(val_df, self.val_csv_name)
        save_df(test_df, self.test_csv_name)
