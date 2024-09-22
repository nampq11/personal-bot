import string
import re
import json
from pydantic import BaseModel

def substitute_punctuation(text):
    translator = str.maketrans(string.punctuation, "_" * len(string.punctuation))
    return text.translate(translator)

def pprint_pydantic_model(model: BaseModel):
    return json.dumps(model.model_dump(), indent=2)