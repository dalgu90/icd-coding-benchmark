import os

import pandas as pd

from src.modules.preprocessors import (ReformatICDCode,
                                       RemoveNumericOnlyTokens, ToLowerCase)
from src.utils.file_loaders import load_csv_as_df


class MimiciiiPreprocessingPipeline:
    def __init__(self, config):
        self.config = config
        self.MIMIC_DIR = config.dirs.mimic_dir
        self.clinical_note_config = config.clinical_note_preprocessing
        self.code_config = config.code_preprocessing

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
            diagnosis_code_df["ICD9_CODE"] = diagnosis_code_df.apply(
                lambda row: str(reformat_icd_code(str(row[4]), True)), axis=1
            )
            procedure_code_df["ICD9_CODE"] = procedure_code_df.apply(
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
        hadm_ids = set(noteevents_df["HADM_ID"])
        code_df = code_df[code_df["HADM_ID"].isin(hadm_ids)]
        return code_df

    def preprocess_clinical_note(self, clinical_note):
        if self.clinical_note_config.lower_case.perform:
            to_lower_case = ToLowerCase()
            clinical_note = to_lower_case(clinical_note)

        if self.clinical_note_config.remove_punc_numeric_tokens.perform:
            remove_numeric_only_tokens = RemoveNumericOnlyTokens()
            clinical_note = remove_numeric_only_tokens(clinical_note)

        return clinical_note

    def preprocess_clinical_notes(self):
        print("Processing Clinical Notes")  # To-do: Add a progress bar
        notes_file_path = os.path.join(
            self.MIMIC_DIR, self.config.dirs.noteevents_csv_name
        )

        noteevents_df = pd.read_csv(notes_file_path)
        # To-do: Add other categories later, based on args provided by the user
        noteevents_df = noteevents_df[noteevents_df["CATEGORY"] == "Discharge summary"]
        # Preprocess clinical notes
        noteevents_df = noteevents_df["TEXT"].apply(self.preprocess_clinical_note)
        # Delete unnecessary columns
        noteevents_df = noteevents_df[["SUBJECT_ID", "HADM_ID", "CHARTTIME", "TEXT"]]
        return noteevents_df

    def combine_code_and_notes(self, code_df, noteevents_df):
        # Sort by SUBJECT_ID and HADM_ID
        noteevents_df = noteevents_df.sort_values(["SUBJECT_ID", "HADM_ID"])
        code_df = code_df.sort_values(["SUBJECT_ID", "HADM_ID"])

        subj_id_hadm_id_list = list(zip(code_df["SUBJECT_ID"], code_df["HADM_ID"]))
        final_df = pd.DataFrame(columns=["SUBJECT_ID", "HADM_ID", "TEXT", "LABEL"])
        for subj_id, hadm_id in subj_id_hadm_id_list:
            code_df_rows = code_df[
                (code_df["SUBJECT_ID"] == subj_id) & (code_df["HADM_ID"] == hadm_id)
            ]
            noteevents_df_rows = noteevents_df[
                (noteevents_df["SUBJECT_ID"] == subj_id)
                & (noteevents_df["HADM_ID"] == hadm_id)
            ]

            codes = []
            notes = []
            for _, row in code_df_rows.iterrows():
                codes.append(row["ICD9_CODE"])
            for _, row in noteevents_df_rows.iterrows():
                notes.append(row["TEXT"])
            final_df.append(
                {
                    "SUBJECT_ID": subj_id,
                    "HADM_ID": hadm_id,
                    "TEXT": " ".join(notes).strip(),
                    "LABEL": ";".join(codes),
                },
                ignore_index=True,
            )
        return final_df

    def preprocess(self):
        code_df = self.extract_df_based_on_code_type()
        noteevents_df = self.preprocess_clinical_notes()
        code_df = self.filter_icd_codes_based_on_clinical_notes(code_df, noteevents_df)
        combined_df = self.combine_code_and_notes(code_df, noteevents_df)
        return combined_df
