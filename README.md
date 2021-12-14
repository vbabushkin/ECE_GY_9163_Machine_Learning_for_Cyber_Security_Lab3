# ECE-GY 9163:  Machine Learning for Cyber Security
# Lab 3

## Vahan Babushkin

### The structure of file directory:

```bash
├── data 
    ├── Lab3
        └── cl
            └── valid.h5     // this is clean validation data used to design the defense
            └── test.h5      // this is clean test data used to evaluate the BadNet
        └── bd
            └── bd_valid.h5 // this is sunglasses poisoned validation data
            └── bd_test.h5  // this is sunglasses poisoned test data
├── models
    └── bd_net.h5      //this is the badnet model used in Lab3
    └── bd_weights.h5  //badnet model weights
├── IMAGES // images in .pdf/.jpg format to test the badnetEval.py script that accepts a test image (in png or jpeg format) and outputs class label in range [0, 1283]
├── REPAIRED_MODELS
    └── bd_repaired_2.h5          //repaired model for 2% accuracy drop
    └── bd_repaired_2_weights.h5  //weights of repaired model for 2% accuracy drop
    └── bd_repaired_4.h5          //repaired model for 4% accuracy drop
    └── bd_repaired_4_weights.h5  //weights of repaired model for 4% accuracy drop
    └── bd_repaired_10.h5          //repaired model for 10% accuracy drop
    └── bd_repaired_10_weights.h5  //weights of repaired model for 10% accuracy drop
└── badnetEval.py  // this is the evaluation script
└── MLSec_Vahan_Babushkin_Lab3_v3.ipynb //jupyter notebook
└── MLSec_Vahan_Babushkin_Lab3_v3..pdf  // lab report
```
The aim of this lab is to design a backdoor detector for BadNets trained on the YouTube Face dataset using the pruning defense. For every image input the backdoor detector outputs the the correct class in range of [0, 1283] if the test input is clean. And it outputs class 1284 if the input is backdoored.

According the project instructions, the modified eval.py script should accept a test image (in png or jpeg format), and output a class in range [0, 1283].

The modified evaluation script (saved as badnetEval.py) accepts a test image (in png or jpeg format) and outputs 1283 if the test image is poisoned, otherwise, if image is clear it outputs the class in range [0,1282]. 

To evaluate the repaired backdoored model (goodnet G) on a test image (in png or jpeg format), execute `badnetEval.py` by running:  
      `python3 badnetEval.py <path to a test image> <repaired model directory>`.
      
E.g., `python3  badnetEval.py  IMAGES/bd/test_0_1024_12819.jpeg  models/bd_net.h5 REPAIRED_MODELS/bd_repaired_10.h5`. 
      
This will output:

      Badnet predicted label:               0
	  Repaired Network predicted label:   969
      Goodnet G predicted label:         1283


We also modified the original script eval.py to read the data in .h5 files and output corresponding class label in range [0, 1283] for each datum. 

To evaluate the repaired backdoored model (goodnet G) on a test image (in png or jpeg format), execute `badnetEval.py` by running:  
      `python3 eval.py <test data directory> <repaired model directory>`.
      
E.g., `python3 eval.py  data/Lab3/cl/test.h5  models/bd_net.h5 REPAIRED_MODELS/bd_repaired_10.h5`. 
      
This will output:

      Badnet classification accuracy: 98.620421
	  Goodnet classification accuracy: 84.544037
	  
	  Badnet predicted label:             950
	  Repaired Network predicted label:   950
	  Goodnet G predicted label:          950
	  Badnet predicted label:             992
	  .......................................
	  
	  Repaired Network predicted label:   872
	  Goodnet G predicted label:          872

In general for this type of backdoor attack the success rate drops sharply when most of the neurons are pruned. However, at the beginning the attack success rate remains around 100% and the clean classification accuracy remains constant. It can be described as follows -- at the beginning we prune neurons which are all zeros or poorly activated, and thus, are not used either by a hones network or badnet. Then when the number of channels removed is above 70%  and below 83% of their initial quantity, we notice drop in the clean classification accuracy while. It means that we are pruning now neuorns that are responsible for classifying the clear inputs but not neuronts, that are activated by the bad inputs. And finally, starting from 83% of all neurons removed both the attack success rate and the clean classification accuracy drop, meaning that now we are now removing those neurons that are both activated by clean and bad inputs. In this case the backdoor attack is disabled, but the clean classification accuracy also drops (e.g. decrease of the attack success rate to 6% results in decline in clean classification accuracy to almost 50%). 

We can notice that the repairing models is not too effective -- in most cases it does not prevent the attack. Only for the repaired network (B') obtained from activations when validation data accuracy drops at least 10% below the original accuracy the success rate is lower compared to the prediction accuracy. The accuracy of Goodnet (G) is slightly lower than of repaired networks (B') since it removes some labels that were misclassifie by badnet. But still the success rate of the attack remains too high, because the repaired badnets (B') still provide 100% success rate. These results suggest that we are dealing with pruning-aware attack, i.e. the attacker recorded the backdoor behavior into the same neurons that are used for classifying the clean data.


In overall, the goodnet drops the clear classification accuracy, which is expected, since it is a pruning-aware attack and an attacker mostly modified those neurons that get activated for a clean data. Therefore, pruning those neurons results in the drop in the clean classification accuracy.

However, pruning the neurons also results in the drop in attack success rate (e.g. from 100% to almost 77% for a repaired model prunned until the validation accuracy dropped below 10% the original accuracy). Therefore, while pruning provides a weak defense for this type of backdoor attack, it still reduces the attack success rate.




