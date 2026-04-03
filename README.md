# research-paper-explainer
This research paper explainer is a tool that helps break down complicated research papers by generating simplified summaries of the sections. The user will upload a pdf, the system will extract the text and use a LLM to produce a summary of the whole paper, generate key points for each section, define terms that are complicated and give the key contributions. The system will also identify technical terms and generate a glossary so researchers and students can refer back to it if needed. Retrieval methods are used so that the model’s explanations remain grounded in the content of the uploaded paper. This tool makes dense research papers more accessible for students and researchers who want to quickly understand the main ideas and results.

**Logic**
This program is an LLM-powered research paper summarizer. The goal is to take a research paper, identify the most important parts, and produce a clear summary that is easier and faster to read than the full paper. It makes the paper easier to read for those who do not understand the techincal jargon. The program pulls out the core ideas such as the research question, methods, results, and conclusions. This program is useful for students and researchers who need to read many paper quickly and efficiently. This can also make a new field or topic sound less daunting as it breaks the paper down into easily digestible parts.

**load_paper(file_path)**
Parameter: file_path(String) -> the path to the input file which contains the text
Purpose: Opens and reads the paper so the rest of the program can process it

**retrieve_sections(paper_text)**
Parameter: paper_text(String) -> this is the full text of the paper
Purpose: Separates the paper into meaningful sections such as title, abstract, introduction, methods, results, and conclusion.

**clean_text(paper_text)**
Parameter: paper_text(String) -> this is the full text of the paper
Purpose: Removes unnecessary formatting, extra spaces, citation clutter, or broken line breaks.

**summarize_paper(paper_text, summary_type="general", max_length=300)**
Parameters: paper_text(String) -> this is the full text of the paper
            summary_type(String) -> this is the type of summary the reader wants. examples: "general", "sections", "results", "simple"
            max_length(int) -> how long the summary should be
Purpose: Generates the summary using the language model.

**save_summary(summary_text, output_path)**
Parameters: summary_text(String) -> The summary of the paper
            output_path(String) -> where to save it
Purpose: Writes the summary to a file.

**Example Use Cases**
A student might use this program when preparing for an exam or writing a literature review. Instead of reading 20 full papers word-for-word, they could first generate summaries and decide which papers deserve a closer read.
A researcher might use it to keep up with newly published papers in their field. Since academic publishing moves quickly, a summarizer can help them scan papers efficiently and identify the ones most relevant to their work.
Someone new to a topic could also use it as a learning tool. A simplified summary can make unfamiliar research easier to approach before reading the original paper.
As shown above this tool can be used by a variety of different people from those who are researchers or thoe just curious about a topic

**Input Data Structure**
Format: txt file
Required Content:
- The text of the paper, ideally in readable order
- Best if it includes labeled sections such as:
- Title
- Abstract
- Introduction
- Methods
- Results
- Discussion / Conclusion
