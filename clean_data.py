import re
def clean(text):
    text = re.sub(r"\([^)]*\d{4}[^)]*\)", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\$.*?\$", "", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = re.sub(r"[^a-zA-Z0-9.,!? ]", "", text)
    text = " ".join(text.split())
    return text.strip()

cleaned_text = []

with open("data/raw_data.txt", "r", encoding="utf-8") as fh:
    for line in fh:
        cleaned = clean(line)
        if len(cleaned) > 50:
            cleaned_text.append(cleaned)

with open("data/clean_data.txt", "w", encoding="utf-8") as fh:
    for l in cleaned_text:
        fh.write(l + "\n")

print("Finished cleaning data")