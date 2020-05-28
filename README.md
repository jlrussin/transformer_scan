# Transformers and systematicity 

Final project for UC Davis computational linguisitcs course in Spring 2020. 

Experiments using transformer models to test systematic/compositional generalization with SCAN dataset.


### References
Transformer model:https://arxiv.org/abs/1706.03762

SCAN dataset: https://arxiv.org/abs/1711.00350

### Repository structure
* transformer_scan
  * data (all data used in experiments)
    * scan (SCAN task)
      * simple (random split)
      * addjump (systematic generalization split)
  * models
    * transformer (PyTorch code implementing transformer model)
  * results (training and testing results)
  * scripts (scripts for running jobs on gpus)
  * main.py (main function - mostly just gathers arguments and runs train())
  * train.py (main training function)
  * data.py (code for dealing with SCAN dataset)
  * test.py (function used for testing and computing accuracy)
  * results.ipynb (notebook for displaying results in results/)

### To run:
Simple split:
```
python main.py --split simple --out_data_file train_defaults_simple
```

Add-jump split:
```
python main.py --split addjump --out_data_file train_defaults_addjump
```
