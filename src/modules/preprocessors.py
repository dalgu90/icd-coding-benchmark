from nltk.tokenize import RegexpTokenizer

from src.modules.tokenizers import *
from src.modules.embeddings import *
from src.utils.mapper import ConfigMapper


# Clinical Note preprocessing

# Ref.: CAML
class ToLowerCase():
    def __init__(self):
        pass

    def __call__(self, text):
        return text.lower()

# Ref.: CAML
# Remove punctuation and numeric-only tokens, removing 500 but keeping 250mg
class RemoveNumericOnlyTokens():
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')
        
    def __call__(self, text):
        tokens = [t for t in self.tokenizer.tokenize(text) if not t.isnumeric()]
        text = '"' + ' '.join(tokens) + '"'
        return text

# ICD-code preprocessing


# Put a period in the right place because the MIMIC-3 data files exclude them.
# Generally, procedure codes have dots after the first two digits, 
# while diagnosis codes have dots after the first three digits.
class ReformatICDCode():
    def __init__(self):
        pass

    def __call__(self, icd_code, is_diagnosis_code):
        code = ''.join(icd_code.split('.'))
        if is_diagnosis_code:
            if code.startswith('E'):
                if len(code) > 4:
                    code = code[:4] + '.' + code[4:]
            else:
                if len(code) > 3:
                    code = code[:3] + '.' + code[3:]
        else:
            code = code[:2] + '.' + code[2:]
        return code
