
## RN4NMT: Refining Source Representations with Relation Networks for Neural Machine Translation

Relation Networks (RNs) is employed to associate source words with each other so that the source representation can memorize all the source words and also contain the relationship between them. Then the source representations and all the relations are fed into the attention component together while decoding, with the main encoder-decoder Neural Machine translation (NMT) architecture unchanged.

> Wen Zhang, Jiawei Hu, Yang Feng and Qun Liu. Refining Source Representations with Relation Networks for Neural Machine Translation. In Proceedings of Coling, 2018. [\[paper\]](https://arxiv.org/pdf/1805.11154.pdf)[[code]](https://github.com/zhang-wen/RN4NMT)

### Runtime Environment
This system has been tested in the following environment.
+ 64bit-Ubuntu
+ Python 2.7
+ Pytorch 0.3.1
+ Dependency
	+ download [Standford parser Version 3.8.0](https://nlp.stanford.edu/software/stanford-parser-full-2017-06-09.zip)
	+ unzip
	+ ``export CLASSPATH="./stanford-parser-full-2017-06-09/stanford-parser.jar:$CLASSPATH"``

### Toy Dataset
+ The training data consists of 44K sentences from the tourism and travel domain
+ Validation Set was composed of the ASR devset 1 and devset 2 from IWSLT 2005
+ Testing dataset is the IWSLT 2005 test set.

### Data Preparation
Name the file names of the datasets according to the variables in the ``wargs.py`` file  
Both sides of the training dataset and the source sides of the validation/test sets are tokenized by using the Standford tokenizer.

#### Training Dataset

+ **Source side**: ``dir_data + train_prefix + '.' + train_src_suffix``  
+ **Target side**: ``dir_data + train_prefix + '.' + train_trg_suffix``  

#### Validation Set

+ **Source side**: ``val_tst_dir + val_prefix + '.' + val_src_suffix``    
+ **Target side**:  
	+ One reference  
``val_tst_dir + val_prefix + '.' + val_ref_suffix``  
	+ multiple references  
``val_tst_dir + val_prefix + '.' + val_ref_suffix + '0'``  
``val_tst_dir + val_prefix + '.' + val_ref_suffix + '1'``  
``......``

#### Test Dataset
+ **Source side**: ``val_tst_dir + test_prefix + '.' + val_src_suffix``  
+ **Target side**:  
``for test_prefix in tests_prefix``
	+ One reference  
``val_tst_dir + test_prefix + '.' + val_ref_suffix``  
	+ multiple references  
``val_tst_dir + test_prefix + '.' + val_ref_suffix + '0'``  
``val_tst_dir + test_prefix + '.' + val_ref_suffix + '1'``  
``......``
 
### Training
Before training, parameters about training in the file ``wargs.py`` should be configured  
run ``python _main.py``

### Inference
Assume that the trained model is named ``best.model.pt``  
Before decoding, parameters about inference in the file ``wargs.py`` should be configured  
+ translate one sentence  
run ``python wtrans.py -m best.model.pt``
+ translate one file  
	+ put the test file to be translated into the path ``val_tst_dir + '/'``  
	+ run ``python wtrans.py -m best.model.pt -i test_prefix``








