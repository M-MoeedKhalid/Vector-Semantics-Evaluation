# Vector-Semantics-Evaluation
A model written to check if vector semantics methods can capture lexical semantics among different words. We compare two ofour gold standards i.e. Simlex-999 and Wordsim-353 with differenct baseline methods. This was done to provide an insight on if vector semanticmethods capture lexical semantics along the way and if they dothen to what degree.

Given a golden standard G and a large corpus of text C for English language, calculate the average Information Retrieval (IR) metricm of top-k similar words retrieved by the vector semantics basedon method V.

Of the 999 words in our simlex-999 dataset only 438 of them were present in the news corpus and out of those438 only 12 of them showed overlapping between the vectors detected by our w2v and the top k values that were obtained via our word2vec model.

The MAP and nDCG are inthe order of 10^−2, which makes us certain that for this particular dataset and trained word2vec vector semantics do a very poor job of capturing lexical semantics with them.

To run this code:
`pip install -r requirements.txt`
Manually install pytrec_eval
`python vector_semantics_evaluation.py`
The tf-idf portions of the file are commented out with comments of which code is using what library.
