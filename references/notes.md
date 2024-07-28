
## TODO

- Analysis. Include just-quantization in comparison. 8 bits
- Experiments. Run for all depth optimization strategies. Verify showing same tendencies.
- Setup full evaluation pipeline for HAR
- Setup model export and microcontroller firmware
- Investigate hyperparameter for controlling leaf clustering

Rename the experiments. hard, soft-f32, soft-u8, cluster-u8, soft-u4, cluster-u4 ?

## Open questions


#### What range of models are feasible at all

model size total. `<< 1MB`, preferably 10-100 kB

Plot model size estimates for datasets.
With majority voting, best case scenario.

median around 10kB.
Majority under 100kB. OK.

#### What differences are there between soft and hard voting

Thesis:
1. As the number of trees get smaller (to reduce model size), the importance of limiting depth goes up (to preserve performance / reduce overfitting)
2. As the number of trees, and depth is restricted - performance drop due to majority vote goes up
=> potential for restoring performance by clustered leaves


BASELINE. scikit-learn defaults. 100 trees, no limit.

Consider only a simple approach of breadth+depth.
Using n_estimators and min_samples_leaf
Keeping max_features to default.

5-40 trees. min_samples_leaf. 1-128, power of 2

### Hyperparameter selection strategy.

As number of trees get lower, need more regularization.
But also have to ensure that each tree is strong enough.
n_features = sqrt(features) can be problematic, because lower chance to hit the meaningful features.
Select informative features first?
Dropping uninformative ones probably valuable, at least.
Wide range of number of features in dataset.
From a few up to 250.


#### How well does feature quantization work

Hypothesis: 16 bit feature is ~lossless.

Results when running separate training runs indicate that 16 bit features is indeed lossless
Was within the margins of error for experiment (rather large because of random differences).

Can be improved by doing a post-training quantization?
Would need to scale the values to be int16 compatible before training.
Then apply feature quantization for test set only.

#### How well does leaf quantization work

Hypothesis: A 8 bit leaf probabilities is ~lossless
Seems to be the case.

Not needed to analyze separately from clustering?
Seems that most of the gains in model size comes from the 32bit->8bit storage for leaf values.
TODO: should also run analysis with plain 8-bit quantization.
Maybe also 4-bit quantization with or without clustering?

#### What is the potential savings from leaf clustering

Shown for n_samples_leaf.
Assuming the same behavior from other restrictions on depth, such as max_depth.
TODO: run an experiment also with max depth, to confirm this

20,18,16,14,12,10,8,6,4,2
20,16,12,8,4


#### How well does leaf clustering work?

Hypothesis: Can reduce leaf size by 2-10x with ~zero loss in performance. Can reduce overall model size by 2-5x

Preliminary results do indicate that up to 5x model size reduction is possible with low perf drops.
However there are a few outliers, where even 0.8 the original size causes drop in performance?

Pleliminary results indicate that under 8 leaves per class is enough to get within 1% drop on all datasets.


#### What practical gains can one reach when combining the techniques

Compare in terms of size and inference time.

- emlearn (int16, 8bit leaf cluster)
- emlearn 0.1x (float, majority)
- m2cgen (float, ? )
- micromlgen (float, ? )

On HAR datasets.


## Out-of-scope

To keep the paper focused. Candidates for "future work".

#### How does code generation vs data structure compare, wrt size and inference time

Hypothesis: datastructure "loadable" is smaller in size, but slower in inference time

Reason for out of scope.
Inference time is so fast that it does not really matter

#### What are limitating factors for practical models on microcontrollers
Hypothesis 1: Model size is the primary bottleneck/constraint over inference time.

accelerometer. 10 FPS
sound. 25 FPS
images. 1 FPS

Maybe do a "worst case" analysis of largesr ensembles that fit on a micro. 10kB, 100kB.
Execution speed for max depths.
Or do a synthetic N deep execution speed test? To get time-per-decision. On common micros

Hypothesis 2: Feature extraction dominates tree execution
maybe compare tree execution speed with a simple preprocessing. EX: RMS

#### Sub-byte leaf quantization

Possibly 4 bits is enough in many cases?
Half the size or leaves. 16 levels instead of 256.
Might shift optimal to a few more leaves.
As long as not 2x, still an overall gain.

#### Decision node with implicit left node

Could probably go down to 4 bytes per node?

- uint8 feature, 255==leaf.
- uint8 right node
- int16 threshold.

Limits right side jumps to 256 steps.
More than that, and need to inject a dummy "jump" node?
Is this case actually needed in practice? Under what conditions.

And will get one extra node per leaf.
Worst case of 50% leaf nodes. Still a win, when node is 4 bytes intead of 2.
Expected model size to be 50% to 75% of 8 bytes per node.

Actually 31 bits of space for in-line leaf value.
Up to 7 classes with 4 bits quantization.
Or 4 classes with 7 bits quantization.
A bit limited when more classes are needed?

Could  be a more custom byte-stream,
where a leaf node can take 1 or more node slots, in case needed to support more leafbits*classes.
First few bits would need to encode how many slots are needed.
Maybe keep it 32 bit aligned for simplicity.
Basically just an "inline" representation for leaf nodes,
instead an index/pointer to external array.

#### Struct of arrays

Alternative is to move to an struct-of-array instead of array-of-structs.
Avoids the (32-bit) padding issue.
Easy to support either 8 or 16 bit thresholds.

- feature. uint8, 255==leaf. Use both left+right to encode leaf pointer.
- right childs. uint8 
- left childs. uint8 
- thresholds. int16 or int8

5 bytes per decision node for int16. Or 4 bytes for int8.
Expected model size being 60% the size of now? 1.8x the efficiency. Not reducing the leaves.
Same structure with external leaf array.
Flexible wrt number of classes. Or regression.
Still making use of leaf clustering.

Questions. Less cache friendly than now?
If inference time goes down, maybe still worth it for the savings in model size?

There is essentially 1-1.2 leaf nodes per decision node.
So for decision+leaf, it is hard to do better than 48 bits - 6 bytes, if having 16 bit features and reach for leaves.
Leaf pointer can be 16 bits.
16 bits for threshold
8 bits for feature
8 bits for child
Using 8 bits for threshold and leaf allow 32 bits. But will not work for all sized forests.

So for 16 bit features and 15 bits leaf reach, 8 bytes (incl padding) per decision node, with leaf indices included, is pretty good.

## Expertiments

C. Need to select number of leaf clusters
Q. Can good values for number of leaf clusters be inferred from the pre-clustered leaf values?

Try to measure the RMSE/MAE of cluster values wrt "perfect" single-class values. 1.0 0.0 1.0
Also measure the RMSE/MAE after clustering?
Plot error metrics vs performance reduction, for the various n leaf clusters

## Dataset selection

Our model system has limited number of features.
n_features <= 255

? how many features are in the OpenML datasets

The following have more than 255.
Most are image recognition. Not so relevant without feature engineering.
har is the most relevant. But that can be studied in next case.

semeion. 
madelon. artificial dataset
har
isolet. Spoken digits
mnist_784
Fashion-MNIST
cnae-9. text recognition.
Devnagari-Script
Internet-Advertisements. mixed
Bioresponse
CIFAR_10


Dropping tasks not fitting criteria
          tid    did                     name
13         15     15                 breast-w
24         29     29          credit-approval
591      2079    188               eucalyptus
1048     3021     38                     sick
1087     3481    300                   isolet
1169     3573    554                mnist_784
1494     3904   1053                      jm1
3418     7592   1590                    adult
3441     9910   4134              Bioresponse
3494     9964   1501                  semeion
3505     9976   1485                  madelon
3510     9981   1468                   cnae-9
4695    14954   6332           cylinder-bands
4708    14970   1478                      har
8678   125920  23381            dresses-sales
11955  146800  40966              MiceProtein
11978  146825  40996            Fashion-MNIST
14839  167121  40923         Devnagari-Script
14841  167124  40927                 CIFAR_10
14842  167125  40978  Internet-Advertisements

52 datasets total.


## Publication target

What would be a good venue for publishing this paper?
Journal or conference.
Allow Arxiv preprint.

Machine Learning, efficient / low power.
TinyML.

MDPI Sensors
Communications of the ACM (CACM). https://dl.acm.org/magazine/cacm
IEEE ?


## Aside. Tiny RNN works

Great listings of papers on RNN
https://github.com/gigwegbe/tinyml-papers-and-projects?tab=readme-ov-file

- FastGRNN: A Fast, Accurate, Stable and Tiny Kilobyte Sized Gated Recurrent Neural Network
- Pushing the limits of RNN Compression
- RNN Inference in 2KB of RAM
- COMPRESSING RNNS FOR IOT DEVICES BY 15-38X USING KRONECKER PRODUCTS
- Efficient Non-linear Pooling for RAM Constrained Inference

https://en.wikipedia.org/wiki/Kronecker_product

https://santiag0m.github.io/blog/2021/04/02/why-are-kronecker-products-so-effective.html

https://www.inference.vc/dilated-convolutions-and-kronecker-factorisation/

https://rosettacode.org/wiki/Kronecker_product#C
