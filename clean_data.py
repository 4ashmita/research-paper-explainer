import re
def clean(text):
    text = re.sub(r"\([^)]*\d{4}[^)]*\)", "", text) # Deletes citation with years within abstract
    text = re.sub(r"\[\d+\]", "", text) # Deletes the numbered bracket citations
    text = re.sub(r"\$.*?\$", "", text) # Deletes the Latex math blocks within the abstract 
    text = re.sub(r"\\[a-zA-Z]+", "", text) # Deletes Latex commands like /alpha
    text = re.sub(r"[^a-zA-Z0-9.,!? ]", "", text) # Deletes anything that is not a standard letter, number, space, or basic punctuation
    text = " ".join(text.split())
    return text.strip()

cleaned_text = []

with open("Unit_Test/raw_out_data.txt", "r", encoding="utf-8") as fh:
    for line in fh:
        cleaned = clean(line)
        if len(cleaned) > 50:
            cleaned_text.append(cleaned)

with open("Unit_Test/clean__out_data.txt", "w", encoding="utf-8") as fh:
    for l in cleaned_text:
        fh.write(l + "\n")

print("Finished cleaning data")