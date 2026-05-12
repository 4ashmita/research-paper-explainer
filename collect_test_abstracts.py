import random
import re

# Read the original file
with open('data/summary_pairs.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the text between 'Abstract:' and 'Summary:'
# (.*?) captures the content, (?=Summary:) stops right before the word Summary
abstracts = re.findall(r"Abstract:(.*?)(?=Summary:)", content, re.DOTALL)

# Randomly sample 15
sampled = random.sample(abstracts, 15) if len(abstracts) >= 15 else abstracts

# Clean whitespace
final_list = [f"{a.strip()}" for a in sampled]

# Write to  new file
with open('Unit_test/test_abstracts_training_set.txt', 'w', encoding='utf-8') as f:
    f.write("\n\n".join(final_list))

print(f"Successfully created test_abstracts.txt with {len(final_list)} clean abstracts.")