Zhuoran Liu - zl2621
Homework #4

#=============================================================================#

This dependency parser is implemented with Python 2.7.14 and Theano 1.0.1

All output files are contained in directory ./output/

How to test prediciton files:
python src/eval.py <gold_label_file> output/<corresponding_output_file>

How to run full experiment code:
./run.sh
If you want to use gpu, simply change the script to set DEVICE=gpu0 (assume you want to use the first gpu)
( Warning!!! Running this command will overwrite the trained model. With stochasticity in NN training, e.g. data shuffling, parameter initialization, you may not get the result as reported. The accuracy can fluctuate by 1-2%. )


#=============================================================================#

Part 1

Result on dev set:
Unlabeled attachment score 82.41
Labeled attachment score 78.46


#=============================================================================#

Part 2

Result on dev set:
Unlabeled attachment score 84.57
Labeled attachment score 80.83

The accuracy improves by approximately 2% over Part 1. This is because larger amount of parameters allows for capturing more information.


#=============================================================================#

Part 3

Result on dev set:
Unlabeled attachment score 85.08
Labeled attachment score 81.45

Following the settings in 'A Fast and Accurate Dependency Parser using Neural Networks' (Chen and Manning, 2014), I made the following adaptation:
1. I used a cubic tranfer function g(x)=x**3 ;
2. I used l2 regularization, with lambda set to 1e-8 ;
3. I trained 20 epochs ;
4. All other settings are the same as Part 1 .

Analysis:
1. As stated in the paper, a cubic transfer function is able to better capture the interaction between any 3 different elements from the input layer, which can be any combination of words and/or pos-tags and/or dependency labels.
2. Also, l2 regularization avoid overfitting to some extent, which boosts the model performance.
3. In my implementation of Adam, I used gradient clipping by global l2 norm and dealt with nan/inf problem, which address the possible gradient exploding problem when using a cubic transfer function.
