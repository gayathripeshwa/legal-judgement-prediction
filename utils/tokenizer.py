import re

def tokenize(text):
    return re.findall(r'\b\w+\b', text)
