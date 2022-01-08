import copy
import csv
import os
import pandas as pd

from tqdm.auto import tqdm

from src.modules.preprocessors import ReformatICDCode, ToLowerCase, RemoveNumericOnlyTokens

class PreprocessingPipeline():
    def __init__(self, config):
        self._config = config
        self.MIMIC_DIR = config.mimic_dir

    def extract_csv_based_on_code_type(self, code_type, add_period_in_correct_pos=True):
        # Which kind of codes do we want? [Diagnosis, Procedure, Both]
        columns = ["ROW_ID", "SUBJECT_ID", "HADM_ID","SEQ_NUM", "ICD9_CODE"]
        header = copy.deepcopy(columns)
        diagnosis_code_csv_path = os.path.join(self.MIMIC_DIR, "DIAGNOSES_ICD.csv")
        procedure_code_csv_path = os.path.join(self.MIMIC_DIR, "PROCEDURES_ICD.csv")
        assert code_type in ["diagnosis", "procedure", "both"], "code_type should be one of [\"diagnosis\", \"procedure\", \"both\"]"

        diagnosis_code_df = pd.read_csv(diagnosis_code_csv_path)
        procedure_code_df = pd.read_csv(procedure_code_csv_path)
        if add_period_in_correct_pos:
            reformat_icd_code = ReformatICDCode()
            diagnosis_code_df['ABSOLUTE_CODE'] = diagnosis_code_df.apply(lambda row: str(reformat_icd_code(str(row[4]), True)), axis=1)
            procedure_code_df['ABSOLUTE_CODE'] = procedure_code_df.apply(lambda row: str(reformat_icd_code(str(row[4]), False)), axis=1)
            columns.pop()
            columns += ["ABSOLUTE_CODE"]

        if self._config.code_type == "diagnosis":
            code_df = diagnosis_code_df
        elif self._config.code_type == "procedure":
            code_df = procedure_code_df
        else:
            code_df = pd.concat([diagnosis_code_df, procedure_code_df])
        code_df.to_csv(os.path.join(self.MIMIC_DIR, "ALL_ICD_CODES.csv"),
                       index=False,
                       columns=columns,
                       header=header)
        return code_df

    def preprocess_clinical_note(self, clinical_note):
        if(self._config.lower_case):
            to_lower_case = ToLowerCase()
            clinical_note = to_lower_case(clinical_note)

        if(self._config.remove_punc_numeric_tokens):
            remove_numeric_only_tokens =  RemoveNumericOnlyTokens()
            clinical_note = remove_numeric_only_tokens(clinical_note)

        return clinical_note       

    def preprocess_clinical_notes(self):
        print("Processing Discharge Summaries") # To-do: Add a progress bar
        notes_file_path = os.path.join(self.MIMIC_DIR, "NOTEEVENTS.csv")
        output_file_path = os.path.join(self.MIMIC_DIR, "PREPROCESSED_NOTEEVENTS.csv")

        with open(notes_file_path, 'r') as  notes_file:
            with open(output_file_path, 'w') as preprocessed_notes_file:
                notes_file_reader = csv.reader(notes_file)

                # skip header
                next(notes_file_reader)

                preprocessed_notes_file.write(','.join(["SUBJECT_ID", "HADM_ID", "CHARTTIME", "TEXT"]) + "\n")

                for line in tqdm(notes_file_reader):
                    subject = line[1]
                    hadm_id = line[2]
                    charttime = line[4]
                    category = line[6]

                    # To-do: Add other categories later, based on args provided by the user
                    if category == "Discharge summary":
                        clinical_note = line[10]
                        clinical_note = self.preprocess_clinical_note(clinical_note)
                        preprocessed_notes_file.write(','.join([subject, hadm_id, charttime, clinical_note]) + "\n")
