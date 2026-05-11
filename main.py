from summarizer import load_inference_objects, generate_summary

# 1. Initialize once
model, tokenizer = load_inference_objects("mini_gpt_summarizer.pt")

# 2. Your Test Abstract (or load from test_abstracts.txt)
with open("Unit_Test/test_abstracts_training_set.txt", "r") as fh:
    train_set_abstracts = [line.strip() for line in fh if line.strip()]

print("This is the training set!!")
# 3. Generate!
for abstract in train_set_abstracts:
    raw_output = generate_summary(model, tokenizer, abstract)
    # 4. Clean up formatting
    summary_part = raw_output.lower().split("summary")[-1].strip()
    clean_summary = summary_part.lstrip("is ").lstrip("the ").capitalize()
    if not clean_summary.endswith("."):
        clean_summary += "."
    print(f"Summary: {clean_summary}")
    print()

with open("Unit_Test/clean_out_data.txt", "r") as fh:
    test_set_abstracts = [line.strip() for line in fh if line.strip()]

print("This is the outside set!!")
# 3. Generate!
for abstract in test_set_abstracts:
    raw_output = generate_summary(model, tokenizer, abstract)
    # 4. Clean up formatting
    summary_part = raw_output.lower().split("summary")[-1].strip()
    clean_summary = summary_part.lstrip("is ").lstrip("the ").capitalize()
    if not clean_summary.endswith("."):
        clean_summary += "."
    print(f"Summary: {clean_summary}")
    print()

