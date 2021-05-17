import re
from typing import List

# Generic cases


def replace_digits(text):
    return re.sub(r"\d+(\.\d*)?", "__digit__", text)


def replace_mails(text):
    pat = r"[a-zA-Z0-9_.+-]+@ ?[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

    return re.sub(pat, "__mail__", text)


def replace_dates(text):
    pat = r"\d{1,4}[ -./]+\d{1,4}([ -./]+\d{1,4})?"

    return re.sub(pat, "__date__", text)


def replace_urls(text):
    pat = r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"

    return re.sub(pat, "__url__", text)


# Extra and Specific cases


def replace_dotnet(text):
    pat = r"(?<!\w)(\.net)"

    return re.sub(pat, "dotnet", text, flags=re.IGNORECASE)


def strip_markers(
    text, markers: List[str] = ["__digit__", "__date__", "__url__", "__mail__"]
):

    for m in markers:
        text = re.sub(pattern=r"[\W]{0,}(" + m + "){0,}[\W]{0,}", string=text, repl="")

    return text
