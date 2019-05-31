# GloRE++
Data and code for ACL 2019 paper "Global Textual Relation Embedding for Relational Understanding"

## Data
Our data and pre-trained model can be downloaded via [Dropbox](https://www.dropbox.com/sh/6cgefbqi0ufrxxq/AADeirRqFvO4WRucmTOP3nzTa?dl=0). 
```
-GloREPlus
  |-small_graph --- The filtered relation graph used to train the textual relation embedding in this work.  
  |-raw_data --- The raw distant supervision dataset.   
  |-train_data --- The filtered relation graph with the input format of the embedding model, for reference purpose.  
  |-models --- The pre-trained textual relation embedding models.  
  
```
For the filtered relation graph, we have the following format. The 3 columns are tab-separated:  
```
textual_relation  KB_relation global_co-occurrence_statistics
```
For the raw distant supervision dataset, we have the following format, also tab-separated, where en1_id and en2_id are the freebase entity ids for the subject and object entity:  
```
en1_id  textual_relation  en2_id  global_co-occurrence_count     
```

## Code
### Requirements
python 2.7  
tensorflow 1.6.0

### Usage
```
-code  
  |-hyperparams.py ---  Hyperparamter settings and input/output file names.  
  |-train.py --- Train textual relation embedding model.  
  |-eval.py --- Generate textual relation embedding with pre-trained model.  
  |-model.py --- The Transformer embedding model.  
  |-baseline.py --- The RNN embedding model.  
  |-data_utils.py --- Utility functions.  
  
-data  
  |-kb_relation2id.txt --- Target KB relations.  
  |-vocab.txt --- Vocabulary file.  
  |-test.txt --- A sample textual relation input file.  
  
-model  
  to store the pre-trained models.  
  
-result  
  to store the result embeddings.  
```

Create two directories named ```model``` and ```result```, to store pre-trained models and the result textual relation embeddings, respectively.   
```
$ mkdir model  
$ mkdir result
```

Put the pre-trained embedding model under ```model/``` directory. Change the ```model_dir``` variable in ```hyperparams.py``` to the name of the pre-trained model you want to use.  
Prepare the input textual relations (parsed with universial dependency) in a single file, with one textual relation per line. The tokens (including both lexical words and dependency relations) are seperated by "##". Refer to a sample input file ```data/test.txt```.    

Put the the input textual relations file under ```data/``` directory as test.txt. Specify your output file name as the ```output_file``` variable in ```hyperparams.py```. Then run ```eval.py``` to produce embeddings of the input textual relations:  
```
$ python eval.py  
```
The output file of the textual relation embeddings have the similar format as word2vec.  
