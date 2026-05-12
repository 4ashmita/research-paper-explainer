import arxiv
import os

os.makedirs("data",exist_ok=True) # Makes the data directory if it does not exist already
queries = ["cat:cs.CL","cat:cs.LG","cat:q-bio.NC","cat:stat.ML"] # arXiv codes for category

total_target = 5000 # Total number of abstracts
per_query = total_target // len(queries) # How many abstracts from each query

client = arxiv.Client(page_size=100,delay_seconds=3.0,num_retries=3) # Creates a "browser instance"
abstracts = []

for q in queries: # Loops through the queries
    search = arxiv.Search(query=q, max_results=per_query, sort_by=arxiv.SortCriterion.SubmittedDate) # Searches for the abstracts that match the query 
    count = 0
    for r in client.results(search): # Goes through each query
        abstract = r.summary.strip().replace("\n", " ") # Gets rid of the newlines making the abstract one line each
        abstract = " ".join(abstract.split()) # Turns multiple spaces, tabs or formatting into the standard singel space
        if len(abstract) > 80: # Only take abstracts that have a length longer than 80 for quality
            abstracts.append(abstract)
            count += 1
    print(f"I collects {count} abstracts from {q}")

seen = set()
unique_abstract = []
for text in abstracts: # Loops through all the abstracts
    if text not in seen: # Only chooses each unique abstract not letting two abstracts be in the data
        seen.add(text)
        unique_abstract.append(text)

# Writes the abstracts into the file
with open("data/raw_data.txt", "w", encoding="utf-8") as fh:
    for abs in unique_abstract:
        fh.write(abs + "\n")
print(f"Saved {len(unique_abstract)} abstracts to the raw_data file in the data folder")