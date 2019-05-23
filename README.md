# GloRE++
Data and code for ACL 2019 paper "Global Textual Relation Embedding for Relational Understanding"

## Data

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
  |-kb_relation2id.txt --- Target relations.  
  |-vocab.txt --- Vocabulary file.  
  |-test.txt --- A sample textual relation input file.  
```

Create two directories named ```model``` and ```result```, to store pre-trained models and the result textual relation embeddings, respectively.   
```
$ mkdir model  
$ mkdir result
```

Download our pre-trained textual relation embedding models from (), and put under ```model/``` directory. Change the ```model_dir``` variable in ```hyperparams.py``` to the name of the pre-trained model you want to use.  
Prepare the input textual relations in a single file, with one textual relation per line. The words and dependency relations in the textual relation are seperated by "##". Refer to a sample input file ```data/test.txt```. We use universal dependency. Put the the input textual relations file under ```data/``` directory as test.txt. Specify your output file name as the ```output_file``` variable in ```hyperparams.py```. Then run ```eval.py``` to produce embeddings of the input textual relations:  
```
$ python eval.py  
```
The output textual relation embeddings have the same format as word2vec.  
