Zhuoran Liu - zl2621
Homework #2


#===============================================================#

Code structure:
.
├── cfg.counts 	# temporary file
├── cfg.rare.counts 	# temporary file
├── count_cfg_freq.py 	# given file
├── eval_parser.py 	# given file
├── parse_dev.dat 	# given file
├── parse_dev.key 	# given file
├── parser.py 	# required file
├── parse_train.dat 	# given file
├── parse_train.RARE.dat 	# required file
├── parse_train_vert.dat 	# given file
├── parse_train_vert.RARE.dat 	# required file
├── pretty_print_tree.py 	# given file
├── Q4 	# core code for Question 4
│   ├── __init__.py
│   ├── __init__.pyc
│   ├── Main.py
│   └── Main.pyc
├── Q5 	# core code for Question 5 and Question 6
│   ├── __init__.py
│   ├── __init__.pyc
│   ├── Main.py
│   └── Main.pyc
├── q5_eval.txt 	# required file
├── q5_prediction_file 	# required file
├── q6_eval.txt 	# required file
├── q6_prediction_file 	# required file
├── Readme.txt 	# required file
└── run.sh 	# Script that runs all experiments

Running Q5 and Q6 will print the parsing process to standard output.
Normally Q5 takes ~20 sec, and Q6 takes ~40 sec.



#===============================================================#

Question 5:

1. Performace statistics:
+--------------------------------------------------------------+
      Type       Total   Precision      Recall     F1 Score
===============================================================
         .         370     1.000        1.000        1.000
       ADJ         164     0.827        0.555        0.664
      ADJP          29     0.333        0.241        0.280
  ADJP+ADJ          22     0.542        0.591        0.565
       ADP         204     0.955        0.946        0.951
       ADV          64     0.694        0.531        0.602
      ADVP          30     0.333        0.133        0.190
  ADVP+ADV          53     0.756        0.642        0.694
      CONJ          53     1.000        1.000        1.000
       DET         167     0.988        0.976        0.982
      NOUN         671     0.752        0.842        0.795
        NP         884     0.622        0.521        0.567
    NP+ADJ           2     0.286        1.000        0.444
    NP+DET          21     0.783        0.857        0.818
   NP+NOUN         131     0.641        0.573        0.605
    NP+NUM          13     0.214        0.231        0.222
   NP+PRON          50     0.980        0.980        0.980
     NP+QP          11     0.667        0.182        0.286
       NUM          93     0.984        0.645        0.779
        PP         208     0.602        0.639        0.620
      PRON          14     1.000        0.929        0.963
       PRT          45     0.957        0.978        0.967
   PRT+PRT           2     0.400        1.000        0.571
        QP          26     0.647        0.423        0.512
         S         587     0.629        0.785        0.698
      SBAR          25     0.091        0.040        0.056
      VERB         283     0.683        0.799        0.736
        VP         399     0.559        0.594        0.576
   VP+VERB          15     0.250        0.267        0.258

     total        4664     0.714        0.714        0.714
+--------------------------------------------------------------+

2. Observation:
	a) Function words has higher F1-scores than content words. All categories that achieve score higher than 80% are function words (ADP, CONJ, DET, PRON, PRT, and end-of-sentence symbols).
	This is intuitive, because the number of function words is limited while they have higher frequency of appearance, making the estimation of these parameters more accurate.
	b) Rare constituents have worse performance.
	This is because constituents that are rarely seen would lead to inaccurate estimate of parameters. But one exception is when this category of constituent derives function words, because the relatively high frequency of function words mitigates sparsity to some extent.
	c) 'SBAR' has an unreasonably low F1-score.
	This is probably because subordinate clause involves long-distance dependencies, which are hard to capture.



#===============================================================#

Question 6:

1. Performace statistics:
+--------------------------------------------------------------+
      Type       Total   Precision      Recall     F1 Score
===============================================================
         .         370     1.000        1.000        1.000
       ADJ         164     0.689        0.622        0.654
      ADJP          29     0.324        0.414        0.364
  ADJP+ADJ          22     0.591        0.591        0.591
       ADP         204     0.960        0.951        0.956
       ADV          64     0.759        0.641        0.695
      ADVP          30     0.417        0.167        0.238
  ADVP+ADV          53     0.700        0.660        0.680
      CONJ          53     1.000        1.000        1.000
       DET         167     0.988        0.994        0.991
      NOUN         671     0.795        0.845        0.819
        NP         884     0.617        0.548        0.580
    NP+ADJ           2     0.333        0.500        0.400
    NP+DET          21     0.944        0.810        0.872
   NP+NOUN         131     0.610        0.656        0.632
    NP+NUM          13     0.375        0.231        0.286
   NP+PRON          50     0.980        0.980        0.980
     NP+QP          11     0.750        0.273        0.400
       NUM          93     0.914        0.688        0.785
        PP         208     0.623        0.635        0.629
      PRON          14     1.000        0.929        0.963
       PRT          45     1.000        0.933        0.966
   PRT+PRT           2     0.286        1.000        0.444
        QP          26     0.650        0.500        0.565
         S         587     0.704        0.814        0.755
      SBAR          25     0.667        0.400        0.500
      VERB         283     0.790        0.813        0.801
        VP         399     0.663        0.677        0.670
   VP+VERB          15     0.294        0.333        0.312

     total        4664     0.742        0.742        0.742
+--------------------------------------------------------------+

2. Observation:
	a) F1-scores of verbs (verb phrases), and 'SBAR' significantly improves.
	This is probably verb phrases and subordinate clauses usually involves long-distance dependencies. Vertical markovization utilized syntactic information from parent nodes, which would help capturing dependency at a higher level.
	b) Performance of some compound constituents decrease: 'ADVP+ADV', 'NP+ADJ'.
	This is because with the increase of non-terminals, some already rare constituent categories become even more sparse, reducing the accuracy of parameter estimation.
	Also, some constituents have a bare several instances in test corpus, which makes their F1-score fluctuates greatly in different models.

3. Selected example:

	Original sentence:
	But not everybody was making money .

	Gold standard:
	["S", ["CONJ", "But"], ["S", ["NP", ["ADV", "not"], ["NOUN", "everybody"]], ["S", ["VP", ["VERB", "was"], ["VP", ["VERB", "making"], ["NP+NOUN", "money"]]], [".", "."]]]]

	Without vertical markovization:
	["S", ["CONJ", "But"], ["S", ["VP", ["ADV", "not"], ["VP", ["VERB", "everybody"], ["VP", ["VERB", "was"], ["VP", ["VERB", "making"], ["NP+NOUN", "money"]]]]], [".", "."]]]

	With vertical markovization:
	["S", ["CONJ", "But"], ["S", ["NP^<S>", ["ADV", "not"], ["NOUN", "everybody"]], ["S", ["VP^<S>", ["VERB", "was"], ["VP^<VP>", ["VERB", "making"], ["NP^<VP>+NOUN", "money"]]], [".", "."]]]]

	This example shows that with syntactic information from the second "S", its children correctly recognizes 'everybody' as a noun.
	Without markovization, because 'VP --> ADV VP' is more often seen than 'NP --> ADV NOUN' (we usually see something like 'NP --> ADJ NOUN'), the parser went wrong.
	But with vertical markovization, 'everybody' knows it is part of a constituent at the start of a sub-sentence, which can rarely be a verb-phrase. Therefore it got the part right.