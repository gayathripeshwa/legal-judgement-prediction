import re

def preprocess_text(text):
    # Lowercase and tokenize using word boundaries
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens
