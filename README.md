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

We provide pre-trained textual relation embedding models in model/ directory. To use, prepair the input textual relations in a single file, with one textual relation per line. The words and dependency relations in the textual relation are seperated by "##". Refer to a sample input file data/test.txt. We use universal dependency. Put the formatted 
