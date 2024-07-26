
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

#### How well does leaf clustering work?

Hypothesis: Can reduce leaf size by 2-10x with ~zero loss in performance. Can reduce overall model size by 2-5x

Preliminary results do indicate that up to 5x model size reduction is possible with low perf drops.
However there are a few outliers, where even 0.8 the original size causes drop in performance?

Pleliminary results indicate that under 8 leaves per class is enough to get within 1% drop on all datasets.


#### What practical gains can one reach when combining all the methods

Compare emlearn (with leaf clustering) with
m2cgen and micromlgen
In terms of size and inference time.


## Out-of-scope

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
