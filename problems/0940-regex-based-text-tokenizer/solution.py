import re

def tokenize_text(text: str) -> list:
    """
    Split raw text into tokens using regex-based splitting on whitespace
    and punctuation. Returns a list of non-empty stripped tokens.
    """
    tokens = re.findall(r"\w+|[^\w\s]+", text)
    return [p for p in tokens if p]