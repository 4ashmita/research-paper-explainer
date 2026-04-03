def load_paper(file_path):
    # Parameter: file_path(String) -> the path to the input file which contains the text 
    # Purpose: Opens and reads the paper so the rest of the program can process it

def retrieve_sections(paper_text):
    # Parameter: paper_text(String) -> this is the full text of the paper 
    # Purpose: Separates the paper into meaningful sections such as title, abstract, introduction, methods, results, and conclusion.

def clean_text(paper_text):
    #Parameter: paper_text(String) -> this is the full text of the paper 
    # Purpose: Removes unnecessary formatting, extra spaces, citation clutter, or broken line breaks.

def summarize_paper(paper_text, summary_type="general", max_length=300):
    # Parameters: paper_text(String) -> this is the full text of the paper 
    #             summary_type(String) -> this is the type of summary the reader wants. examples: "general", "sections", "results", "simple" 
    #             max_length(int) -> how long the summary should be 
    # Purpose: Generates the summary using the language model.

def save_summary(summary_text, output_path):
    #Parameters: summary_text(String) -> The summary of the paper 
    #            output_path(String) -> where to save it 
    # Purpose: Writes the summary to a file.