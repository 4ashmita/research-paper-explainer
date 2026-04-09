import arxiv
import os

os.makedirs("data",exist_ok=True)
queries = ["cat:cs.CL","cat:cs.LG","cat:q-bio.NC","cat:stat.ML"]

total_target = 3000
per_query = total_target // len(queries)

client = arxiv.Client(page_size=100,delay_seconds=3.0,num_retries=3)
abstracts = []

for q in queries:
    search = arxiv.Search(query=q, max_results=per_query, sort_by=arxiv.SortCriterion.SubmittedDate)
    count = 0
    for r in client.results(search):
        abstract = r.summary.strip().replace("\n", " ")
        abstract = " ".join(abstract.split())
        if len(abstract) > 80:
            abstracts.append(abstract)
            count += 1
    print(f"I collects {count} abstracts from {q}")

seen = set()
unique_abstract = []
for text in abstracts:
    if text not in seen:
        seen.add(text)
        unique_abstract.append(text)

with open("data/raw_data.txt", "w", encoding="utf-8") as fh:
    for abs in unique_abstract:
        fh.write(abs + "\n")
print(f"Saved {len(unique_abstract)} abstracts to the raw_data file in the data folder")