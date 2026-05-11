# research-abstract-explainer
This project focuses on building a small tokenizer-based model that can interpret scientific language and translate it into everyday, accessible language. The model will be trained on a dataset of research paper abstracts and will be able to take in new abstracts as input and generate simplified summaries. The data will be taken from arxiv a huge dataset with research abstracts

**Logic**

This project develops an LLM-powered research paper summarizer that combines a custom-built tokenizer with the GPT-4 model. The system is designed to process scientific abstracts, identify their most important ideas, and generate concise summaries that are easy for a general audience to understand. A key component of the project is the tokenizer, which is built from scratch to better handle domain-specific scientific terminology and structure. By translating complex language into everyday phrasing, the program aims to make academic research more accessible while preserving the core meaning and insights of the original text.

**load_paper(file_path)**

Parameter: file_path(String) -> the path to the input file which contains the text
Purpose: Opens and reads the paper so the rest of the program can process it

**clean_text(paper_text)**

Parameter: paper_text(String) -> this is the full text of the paper
Purpose: Removes unnecessary formatting, extra spaces, citation clutter, or broken line breaks.

**summarize_paper(paper_text, max_length=300)**

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

**Unit Tests**
I will be providing a test_data.txt file with a bunch of abstracts that can be used to test the gpt-4 model. One could also test the project by providing their own abstracts that they would want tested.