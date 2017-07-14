DeepGerman project is an attempt to infer the gender (masculine, feminine, or neutral) of a German noun from its raw character-level representation. The project is implemented using TensorFlow. Several RNN and MLP models of different architecture were trained and evaluated manually (by entering "words") and automatically (by randomly generated "words"). RNN input is one-hot representation of subsequent word characters at each time step. MLP input is concatenated one-hot vectors of all characters. The output of both model types is three-class softmax for three genders.

Runnable scripts:
* **rnn_deep_german.py** - configurable RNN trainer (CL arguments specified in the script).
* **mlp_deep_german.py** - configurable MLP trainer (CL arguments specified in the script).
* **evaluate_manual.py** - manual model evaluation by entering a word and observing the inferred gender probabilities.
* **evaluate_auto.py** - automatic model evaluation by generating multiple random words of variable length with fixed endings corresponding to a given gender (e.g. "-ung" for feminine) and observing the statistics of inferred gender classes for each ending.

The model achieving 3.76% error (2-layer LSTM with 128 hidden units in each layer, trained with dropout and batch size of 128) is available in "results/models" folder. The results of automatic evaluation of this model (obtained by running **evaluate_auto.py**) are shown below. 10,000 random words of varying length have been generated per gender ending. The numbers in three columns of the table show the percentage of the words classified as masculine, feminine, and neutral for each of the endings:

~~~~
masculine endings
-------------------------------------
-ant       96.36     1.96      1.68      
-ast       93.41     3.84      2.75      
-er        89.72     2.35      7.93      
-ich       75.00     0.28      24.72     
-eich      89.94     0.15      9.91      
-ig        75.64     0.10      24.26     
-eig       75.27     0.25      24.48     
-ling      53.68     0.06      46.26     
-or        83.17     2.13      14.70     
-us        91.32     2.92      5.76      
-ismus     100.00    0.00      0.00      

feminine endings
-------------------------------------
-anz       22.27     76.52     1.21      
-e         18.22     75.42     6.36      
-ei        28.23     39.34     32.43     
-enz       4.03      93.92     2.05      
-heit      10.33     89.29     0.38      
-ie        6.68      90.36     2.96      
-ik        22.29     67.71     10.00     
-keit      46.30     53.56     0.14      
-schaft    2.46      97.22     0.32      
-sion      0.08      98.62     1.30      
-sis       5.16      68.50     26.34     
-tion      0.78      97.34     1.88      
-tät       6.87      85.96     7.17      
-ung       7.38      91.13     1.49      
-ur        35.58     59.64     4.78      

neutral endings
-------------------------------------
-chen      3.21      0.00      96.79     
-lein      2.53      0.35      97.12     
-en        18.67     0.53      80.80     
-il        13.78     0.16      86.06     
-ing       10.73     2.13      87.14     
-ma        1.56      23.44     75.00     
-ment      7.92      0.11      91.97     
-nis       12.43     21.68     65.89     
-tum       2.80      0.07      97.13     
-um        17.71     0.24      82.05     
~~~~
