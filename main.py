from summarizer import load_inference_objects, generate_summary, save_summary

# Initialize once
model, tokenizer = load_inference_objects("mini_gpt_summarizer.pt")

# Open the test abstracts from the training set
with open("Unit_Test/test_abstracts_training_set.txt", "r") as fh:
    train_set_abstracts = [line.strip() for line in fh if line.strip()]

test_summaries = [] # Stores the summaries
path = "Unit_Test/test_summaries" # Path to save the summaries

print("This is the training set!!")

# Generates the summaries from the training set
for abstract in train_set_abstracts:
    raw_output = generate_summary(model, tokenizer, abstract)
    # Cleans up formatting
    summary_part = raw_output.lower().split("summary")[-1].strip()
    clean_summary = summary_part.lstrip("is ").lstrip("the ").capitalize() # Is and the is most common to come after the prompt and can cause grammatical issues so it is removed lstrip means remove from left
    if not clean_summary.endswith("."):
        clean_summary += "."
    test_summaries.append(clean_summary)

save_summary(train_set_abstracts, test_summaries, path) # Saves the summaries

# Opens the test abstracts
with open("Unit_Test/clean_out_data.txt", "r") as fh:
    test_set_abstracts = [line.strip() for line in fh if line.strip()]

path = "Unit_Test/outside_test_summaries" # Path to save the summaries
outside_summaries = [] # Stores the summaries

print("This is the outside set!!")
# Generates the summaries from the test set
for abstract in test_set_abstracts:
    raw_output = generate_summary(model, tokenizer, abstract)
    # Cleans up formatting
    summary_part = raw_output.lower().split("summary")[-1].strip()
    clean_summary = summary_part.lstrip("is ").lstrip("the ").capitalize()
    if not clean_summary.endswith("."):
        clean_summary += "."
    outside_summaries.append(clean_summary)

save_summary(test_set_abstracts, outside_summaries, path) # Saves the summaries