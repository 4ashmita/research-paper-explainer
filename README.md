# research-paper-explainer
This research paper explainer is a tool that helps break down complicated research papers by generating simplified summaries of the sections. The user will upload a pdf, the system will extract the text and use a LLM to produce a summary of the whole paper, generate key points for each section, define terms that are complicated and give the key contributions. The system will also identify technical terms and generate a glossary so researchers and students can refer back to it if needed. Retrieval methods are used so that the model’s explanations remain grounded in the content of the uploaded paper. This tool makes dense research papers more accessible for students and researchers who want to quickly understand the main ideas and results.

**Extract text from pdf**
This function takes in the research article as a pdf and extracts the text from it.

**Retrieve sections**
This function will retrieve the relevant section based on the query. If the query asks for a full paper summary then this function will not need to do anything as the whole text will be used.

**Generate explanations**
This function will use the LLM to generate the explanation, summaries, or definitions based on the text and query.
