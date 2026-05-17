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
Before running make sure to download the libraries from requirements.txt. You will need to download torch and numpy.

1st. Run train.py and let that go to completion. That may take some time as it is running 40 epochs on 4000 abstracts. 

Note: If you want to see how it works you can run generate.py which is a script to let the first model generate scientific sound sentences. 

2nd. Run fine_tune.py and let that go to completion. This should be slightly faster as it only runs 10 epochs.

3rd. After both train.py ans fine_tune.py have been completed you will see two .pt files. That is the models. To run the unit test all you have to run is main.py which should use the two test files to give summaries and will save them repsectively to test_summaries and outside_test_summaries.

Note: If it seems slow and you want it to run faster, you can go to train_tokenizer.py and decrease the vocab_size from 40000 to 15000. Keep in mind that doing this however will lead to seeing more [UNK] in the output as the model will have IDs for a smaller amount of words. **After you change the vocab_size you will have to run train_tokenizer.py to actually see the change or the train.py and fine_tune.py will use the tokenizer will the larger vocabulary.**

**Code Structure**

* `model.py`: Holds the code for the miniGPT model.

* `tokenizer.py`: Hold the word level tokenizer.

* `train_tokenizer.py`: Used to train the tokenizer on a set vocabulary level.

*  `train.py`: This is the initial training of the model to learn scientific language. This will be run first and should go to completion

* `generate.py`: This tests the trained model on how well it generate scientific language. To test you can change the prompt to something else and see whether the model gives something coherent and in scientific jargon

* `fine_tune.py`: This is the final training where the model from the beginning is once again trained on a different dataset containing abstracts and a summary. This is what is run after the train.py finishes.

* `summarizer.py`: This script is used to generate the summaries given an abstract.

* `main.py`: This is what is run after both models are trained. This opens two files: one contains 15 abstracts randomly collected from the dataset and the other has 24 abstract that were not shown in the training set. It will generate the summaries and save them to two files.

* `collect_test_abstracts.py`: Collects 15 abstracts that will be used to test the model on the training set

* `collect_outside_abstracts.py`: Collects 24 abstracts that are from different topics than the training avstracts which are used to test the model

* `clean_data.py`: Removes citations, latex math blocks, latex commands, and numbered citations from the abstract

* `dataset.py`: This creates a custom dataset pipeline using PyTorch. In machine learning, the raw list of data can't be given for training this gives a blueprint to PyTorch which requires the data to be formatted specifically a blueprint on how to measure the dataset and how to fetch individual samples

* `download_abstract.py`: Collects the 4674 abstracts from the arXiv database. These are used to intially train the model on scientific language

* `tokenizer.pkl`: Contains the vocabulary mappings that the model learns in preparations. It contains both word to token mappings and token to word mappings. It is a static frozen file that is actively used in both initial training and fine tuning

* `Data`: Contains all the data used to train both models

    * `raw_data.txt`: contains 4674 abstracts before processing them
    
    * `clean_data.txt`: Contains the same 4674 abstracts after they have been proccessed and is used to intially train the model
    
    * `summary_pairs.txt`: Contains 1000 abstract summary pairs used to fine tune the model

* `Unit_Test`: Contains all the files necessary to test the model

    * `raw_out_data.txt`: Contains the 24 abstracts collected before processing

    * `clean_out_data.txt`: Contains the 24 abstracts after processing and removing unecessary text

    * `test_abstract_training_set.txt`: Contains the 15 abstracts randomly collected from the training set

    * `test_summaries.txt`: This contains the summaries for the training set

    * `outside_test_summaries.txt`: This contains the summaries for the outside test or stress test