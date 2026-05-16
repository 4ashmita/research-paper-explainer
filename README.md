# research-abstract-explainer
This project focuses on building a lightweight, custom-trained GPT-style transformer that is designed to simplify complex scientific abstracts. I used a vocabulary-optimized 40,000 tokenizer to traing the model to do two things: Understand scientific language and taking an abstract and giving a simplified summary. To train the model on scientific abstracts, I used 4,674 abstracts sourced from arXiv. To fine tune the model to summarize I took 1000 abstracts selected from the original data and created summaries for those 1000 to let the model understand how to go from abstract to summary. The primary goal was to evaluate how a small-scale model (512-dimensional embeddings, 8-head attention) encodes scientific language into everday summaries.

**Logic**

This project develops an LLM-powered research paper summarizer that combines a custom-built tokenizer with the GPT-4 model. The system is designed to process scientific abstracts, identify their most important ideas, and generate concise summaries that are easy for a general audience to understand. A key component of the project is the tokenizer, which is built from scratch to better handle domain-specific scientific terminology and structure. By translating complex language into everyday phrasing, the program aims to make academic research more accessible while preserving the core meaning and insights of the original text.

**Example Use Cases**

This is kind of the base for a future creation of a full research paper LLM. Right now anyone can use this to summarize the abstract of any paper given. Most people read the abstract first to get an overview of the whole project and this would help them understand the abstract without getting confused by the scientific jargon.

**Input Data Structure**

Format: txt file
Required Content:
- The text of the abstract
- You must run the txt file through clean_data.py which will remove unneccesary text such as the citations, latex math blocks, latex commands, and numbered citations.

**Unit Tests**

I have provided two test files. One is called test_abstract_training_set.txt and contains 15 abstracts from the training set. The other is called clean_out_data.txt which contains 24 abstracts collected from arXiv that are not used in the training set. This is a stress-test to observe Schema Over-generalization.

**How To Run**

To run the project there are three steps:
1st. Run train.py and let that go to completion. That may take some time as it is running 40 epochs on 4000 abstracts. 

Note: If you want to see how it works you can run generate.py which is a script to let the first model generate scientific sound sentences. 

2nd. Run fine_tune.py and let that go to completion. This should be slightly faster as it only runs 10 epochs.

3rd. After both train.py ans fine_tune.py have been completed you will see two .pt files. That is the models. To run the unit test all you have to run is main.py which should use the two test files to give summaries and will save them repsectively to test_summaries and outside_test_summaries.

Note: If it seems slow and you want it to run faster, you can go to train_tokenizer.py and decrease the vocab_size from 40000 to 15000. Keep in mind that doing this however will lead to seeing more [UNK] in the output as the model will have IDs for a smaller amount of words. After you change the vocab_size you will have to run train_tokenizer.py to actually see the change or the train.py and fine_tune.py will use the tokenizer will the larger vocabulary.