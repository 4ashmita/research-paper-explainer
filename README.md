# research-abstract-explainer
This project focuses on building a lightweight, custom-trained GPT-style transformer that is designed to simplify complex scientific abstracts. I used a vocabulary-optimized 40,000 tokenizer to traing the model to do two things: Understand scientific language and taking an abstract and giving a simplified summary. To train the model on scientific abstracts, I used 4,674 abstracts sourced from arXiv. To fine tune the model to summarize I took 1000 abstracts selected from the original data and created summaries for those 1000 to let the model understand how to go from abstract to summary. The primary goal was to evaluate how a small-scale model (512-dimensional embeddings, 8-head attention) encodes scientific language into everday summaries

**Logic**

This project develops an LLM-powered research paper summarizer that combines a custom-built tokenizer with the GPT-4 model. The system is designed to process scientific abstracts, identify their most important ideas, and generate concise summaries that are easy for a general audience to understand. A key component of the project is the tokenizer, which is built from scratch to better handle domain-specific scientific terminology and structure. By translating complex language into everyday phrasing, the program aims to make academic research more accessible while preserving the core meaning and insights of the original text.

**load_inference_objects(checkpoint)**

Parameter: checkpoint(String) -> the pt file which saves the model itself
Purpose: Opens and loads the necessary components liek the model and tokenizer

**clean_text(paper_text)**

Parameter: paper_text(String) -> this is the full text of the paper
Purpose: Removes unnecessary formatting, extra spaces, citation clutter, or broken line breaks.

**generate_summary(model, tokenizer, abstract, max_new_tokens=100, temperature=0.7, top_p=0.9)**

Parameters: paper_text(String) -> this is the full text of the paper
            max_length(int) -> how long the summary should be
Purpose: Generates the summary using the language model.

**save_summary(summary_text, output_path)**

Parameters: summary_text(String) -> The summary of the paper
            output_path(String) -> where to save it
Purpose: Writes the summary to a file.

**Example Use Cases**

This is kind of the base for a future creation of a full research paper LLM. Right now anyone can use this to summarize the abstract of any paper given. Most people read the abstract first to get an overview of the whole project and this would help them understand the abstract without getting confused by the scientific jargon

**Input Data Structure**

Format: txt file
Required Content:
- The text of the abstract
- You must run that abstract throw clean_data.py which will remove unneccesary text such as the citations.

**Unit Tests**
I have provided two test files. One is called test_abstract_training_set.txt and contains 15 abstracts from the training set. The other is called clean_out_data.txt which contains 24 abstracts collected from arXiv that are not used in the training set. That is a stress-test to observe Schema Over-generalization.