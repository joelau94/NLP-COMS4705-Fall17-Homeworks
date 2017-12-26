Zhuoran Liu - zl2621
Homework #1

==========================================
Question 4:

1. Performance of baseline tagger (prediction file '4_2.txt')
+----------------------------------------+
		precision 	recall 		F1-Score
Total:	 0.221961	0.525544	0.312106
PER:	 0.435451	0.231230	0.302061
ORG:	 0.475936	0.399103	0.434146
LOC:	 0.147750	0.870229	0.252612
MISC:	 0.491689	0.610206	0.544574
+----------------------------------------+

2. Observation
The baseline tagger predicts a same tag for the same word every time.
This is obvious, because we are always taking the 'y' that maximizes e(x|y) for the same 'x', while no contextual information was taken into consideration.

==========================================
Question 5:

1. Performance of tagger with Viterbi algorithm (prediction file '5_2.txt')
+----------------------------------------+
		precision 	recall 		F1-Score
Total:	 0.774139	0.613724	0.684661
PER:	 0.759053	0.593036	0.665852
ORG:	 0.611855	0.478326	0.536913
LOC:	 0.876289	0.695202	0.775312
MISC:	 0.830065	0.689468	0.753262
+----------------------------------------+

2. Observation
i) Not surprisingly, Viterbi tagger is much better than the baseline tagger, because contextual information is included.

ii) About errors.

Statistics of error of each tag:
For all words, the number of mistake made under each tag is:
I-LOC: 1254
O: 1166
I-PER: 1980
I-MISC: 754
B-MISC: 4
I-ORG: 1678
For infrequent words (frequency < 5), the number of mistake made under each tag is:
I-LOC: 950
O: 976
I-PER: 1874
I-MISC: 576
B-MISC: 2
I-ORG: 1132
As can be seen, most errors are made when trying to predict tags of low-frequency words. The most affected class of Named Entity by infrequent words is 'I-PER'.

Also, a lot of numbers / digits were tagged as some Named Entity while their original tag were 'O'.

==========================================
Question 6:

According to observation ii) in Question 5, I tried to tackle the issue of un-recognized Persons and mis-recognized Numbers. So I divised the following substitution rules:
1. '_containsDigit_' : all infrequent words that contains a digit
2. '_initCap_' : all infrequent words whose first letter is capitalized
3. '_RARE_' : other infrequent words
The infrequent words are defined as words appearing less than a 'minimun frequency', which can take different values.

Now I'll first report some experimental results, then present my observations and conclusions.

***************** Results *****************
(Best performance: F-1 = 76.78%, Experiment 4)

Experiment 1: minimun frequency = 5, using rules NO. 1,2,3
+----------------------------------------+
		precision 	recall 		F1-Score
Total:	 0.721080	0.603271	0.656936
PER:	 0.661290	0.468444	0.548408
ORG:	 0.566570	0.585202	0.575735
LOC:	 0.861130	0.706652	0.776280
MISC:	 0.825356	0.692725	0.753247
+----------------------------------------+

Experiment 2: minimun frequency = 5, using rules NO. 1 (This is the same setting as in Question 5)
+----------------------------------------+
		precision 	recall 		F1-Score
Total:	 0.774139	0.613724	0.684661
PER:	 0.759053	0.593036	0.665852
ORG:	 0.611855	0.478326	0.536913
LOC:	 0.876289	0.695202	0.775312
MISC:	 0.830065	0.689468	0.753262
+----------------------------------------+

Experiment 3: minimun frequency = 2, using rules NO. 1,2,3
+----------------------------------------+
		precision 	recall 		F1-Score
Total:	 0.787024	0.672905	0.725504
PER:	 0.744428	0.545158	0.629397
ORG:	 0.668698	0.656203	0.662392
LOC:	 0.892541	0.769902	0.826698
MISC:	 0.842169	0.758958	0.798401
+----------------------------------------+

Experiment 4: minimun frequency = 2, using rules NO. 1 (Best performance, default setting of HW6.py)
+----------------------------------------+
		precision 	recall 		F1-Score
Total:	 0.826180	0.717248	0.767870
PER:	 0.845144	0.700762	0.766211
ORG:	 0.712739	0.639761	0.674281
LOC:	 0.881472	0.770447	0.822229
MISC:	 0.848965	0.756786	0.800230
+----------------------------------------+

Experiment 5: minimun frequency = 1, using rules NO. 1,2,3
+----------------------------------------+
		precision 	recall 		F1-Score
Total:	 0.890963	0.402293	0.554304
PER:	 0.981967	0.325898	0.489379
ORG:	 0.813268	0.494768	0.615242
LOC:	 0.908978	0.397492	0.553111
MISC:	 0.876106	0.429967	0.576839
+----------------------------------------+

I've also done other experiments with different minimum frequency and different rules combination. But generally, experimental results show the same trend as those reported above.
*******************************************

*************** Observation ***************
1. Adding more rules leads to slightly lower performance (Comparison between experiment 1 and 2, 3 and 4).
2. Filtering out words only seen once is helpful (Comparison between experiment 3 and 5).
But beyond that, the higher 'minimum frequency' you set, the lower performance you'll get (Comparison between experiment 1 and 3, 2 and 4, and other un-reported experiments on different 'minimum_frequency').
*******************************************

*************** Conclusions ***************
I guess there are several reasons why things happened as described in observations:
1. The training set is too small (only 14k+ sentences), so is the vocabulary size. Under such circumstance, if we set 'minimum frequency' to a higher value, we will lose a lot of useful lexical information.
But removing words only seen once is necessary, because they are destabilizing the parameter estimation.
2. For such a small set of training data, introducing more hand-crafted rules means having more perturbance to the original underlying distribution, which is another destabilizing factor for parameter estimation.
3. The training data is homogenous to the development data (If I was right, both training and development set are taken from Reuters News Stories from within 10 days, as the CoNLL-2003 shared-task proposal describes). Therefore they have similar vocabulary and linguistic features, which means preserving more of the original vocabulary in training set would automatically lead to a better fit of the development set.
*******************************************