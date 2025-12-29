
## TODO

- Re-run plots in notebooks with latest results
- Use int16 features as main for the experiments
- Move plotting code from notebooks to scripts
- Add plotting code to a Actions step
- Put complete pareto frontier plots into Appendix 1
- Change or add HAR training script to output multiple models
- Make script to automate the device test runs.
Run X models, X frameworks, on 1 microcontrollers
Generate X different models, of different sizes and for both datasets
- Run framework comparisons on microcontrollers
- Investigate hyperparameter for controlling leaf clustering

Split out emlearn inline plus loadable?
? try to place loadable in

Maybe change to 256 window size on PAMAP2

Rename the experiments. hard, soft-f32, soft-u8, cluster-u8, soft-u4, cluster-u4 ?


## Claims

#### int16 feature is virtually lossless
Minor. Just a stepping-stone for the next parts.

#### Majority voting can result in signficant performance drop
Demonstrates the problem / potential for improvement.

#### Leaf proportions with low-bits quantization (OR clustering) dominates
Pareto optimial wrt A) hard majority voting and B) full float leaf proportions.
When condidering predictive performance and model size.
Assuming that leaf-deduplication is enabled.
! need to show that there is no majority model with more nodes/trees that
performs better, at same or smaller model size.
On majority of the datasets. Ideally all.

? Enough to show this at a single set of trees. Ex=10

#### emlearn (with leaf quant/cluster) dominates
When considering predictive performance and model size.
Ideally also inference time.
When running on microcontroller.

Show on a few datasets.

## Open questions


#### On device performance

PYTHONPATH=.. python -m src.experiments.compare_frameworks ../output/results/har/r_6A1CF5_uci_har.estimator.pickle
west build -p=always -b rpi_pico trees_run/

TODO: use west flash?
```
openocd -f interface/cmsis-dap.cfg -c 'transport select swd' -f target/rp2040.cfg -c "adapter speed 2000" -c 'targets rp2040.core0'
```
Script to catch build size output
Python script to catch USB serial output


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
Result: Seems to be the case.

Results when running separate training runs indicate that 16 bit features is indeed lossless
Was within the margins of error for experiment (rather large because of random differences).

Can be improved by doing a post-training quantization?
Would need to scale the values to be int16 compatible before training.
Then apply feature quantization for test set only.

#### How well does leaf quantization work

Hypothesis: A 8 bit leaf probabilities is ~lossless
Result: Seems to be the case.

Not needed to analyze separately from clustering?
Seems that most of the gains in model size comes from the 32bit->8bit storage for leaf values.
TODO: should also run analysis with plain 8-bit quantization.
Maybe also 4-bit quantization with or without clustering?

#### What is the potential savings from leaf clustering

Shown for n_samples_leaf.
Assuming the same behavior from other restrictions on depth, such as max_depth.

max_depth also shows this behavior.
TODO: run with the rest of the limiters

#### max_depth
`MAX_DEPTH=20,18,16,14,12,10,8,6,4,2`
Could have dropped 20 and 18

#### min_samples_leaf
`MIN_SAMPLES_LEAF=1,2,4,8,16,32,64,128`

#### min_samples_split
`MIN_SAMPLES_SPLIT=2,4,8,16,32,64,128,256`

#### max_leaf_nodes
`MAX_LEAF_NODES=3,10,32,100,320,1000,3200,10000`
Builds the tree in a best-first fashion rather than a depth-first fashion.
5 to 10k was the approximate range for the min_samples_leaf sweep.

? will it get more unique leaves from the start?

','.join(numpy.logspace(0.5, 3.5, num=12).astype(int).astype(str))
'3,5,11,20,38,73,136,256,480,900,1687,3162'

#### min_impurity_decrease
`MIN_IMPURITY_DECREASE=0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50`
Default 0.0

>>> ','.join((f'{s:.2f}' for s in numpy.linspace(0.0, 0.5, 11)))
'0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50'

','.join(numpy.logspace(-4, -1, num=11).round(4).astype(str))
'0.0001,0.0002,0.0004,0.0008,0.0016,0.0032,0.0063,0.0126,0.0251,0.0501,0.1'


gini impurity naturally ranges between 0 and 1.
Gini is the default criteria for classification.

#### ccp_alpha
`
0.0 -> ?
`

To get an idea of what values of ccp_alpha could be appropriate,
scikit-learn provides DecisionTreeClassifier.cost_complexity_pruning_path
that returns the effective alphas and the corresponding total leaf impurities at each step of the pruning process.
Data-dependent method.


#### min_weight_fraction_leaf
Mostly to deal with imbalance?
Interacts with class weights, if set.
Related to min_samples_split, but weighted




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

#### How does code generation vs data structure compare
In terms of program size and inference time

Hypothesis: datastructure "loadable" is smaller in size, but slower in inference time

! On Linux x64 over 10x speedup was seen, with -O3

Using n a 10 tree model on UCI HAR with timebased features.
Running 60*1000 iterations.
MicroMLGen, which also uses an "inline" strategy also does well.
```
METHOD              TIME[us]
emlearn inline           800
micromlgen              1500
emlearn loadable       10500
m2cgen                 10000
```

Is this the case also on microcontrollers?

RP2040, with Zephyr
```
emlearn loadable
test-complete repeats=100 samples=60 time=372800 errors=2 

emlearn inline
test-complete repeats=100 samples=60 time=2100 errors=2 
test-complete repeats=100 samples=60 time=2000 errors=2 

m2cgen
test-complete repeats=1 samples=60 time=31900 errors=3
test-complete repeats=100 samples=60 time=2026700 errors=3 

micromlgen
test-complete repeats=1 samples=60 time=9600 errors=3
test-complete repeats=1 samples=60 time=9700 errors=3
test-complete repeats=100 samples=60 time=9800 errors=3
```

? Was this emlearn loadable with float?
Latest commit was maybe 6c45cc64d73de7cbf043cd11b49ef89b41c4f11c - 0.21.0
But unsure if that was the test results.

Even bigger difference between inline and lodable!
Over 100x!!
Is it the difference between external FLASH and code mem?
Could try to use non-const memory instead to place into RAM

Initial thoughts: Maybe out of scope?
Inference time is so fast that it does not really matter.
But this result is cause for reconsideration.


### m2cgen compile failure


/home/jon/projects/tiny-random-forests/device/trees_run/../code/model_m2cgen.h:5343:115: error: macro "memcpy" passed 8 arguments, but takes just 3
 5343 |                                         memcpy(var19, (double[]){0.0, 0.0, 0.0, 0.0, 1.0, 0.0}, 6 * sizeof(double));
      |                                                                                                                   ^
/data/emlearn/zephyr-sdk-0.16.5/arm-zephyr-eabi/picolibc/include/ssp/string.h:97: note: macro "memcpy" defined here
   97 | #define memcpy(dst, src, len) __ssp_bos_check3(memcpy, dst, src, len)

Should add parens around the data declaration.

https://github.com/BayesWitnesses/m2cgen/pull/593

###

arch_irq_unlock (key=0)
    at /home/jon/projects/tiny-random-forests/device/zephyr/include/zephyr/arch/arm/asm_inline_gcc.h:102
102		__asm__ volatile(
(gdb) 
0x1000c292	102		__asm__ volatile(
(gdb) 


Program stopped.
0x1000c292 in arch_irq_unlock (key=0)
    at /home/jon/projects/tiny-random-forests/device/zephyr/include/zephyr/arch/arm/asm_inline_gcc.h:102
102		__asm__ volatile(
(gdb) 
[rp2040.core0] target not halted
target rp2040.core0 was not halted when step was requested


^C
Program received signal SIGINT, Interrupt.
arch_system_halt (reason=reason@entry=0)
    at /home/jon/projects/tiny-random-forests/device/zephyr/kernel/fatal.c:32
32		for (;;) {
(gdb) bt
#0  arch_system_halt (reason=reason@entry=0)
    at /home/jon/projects/tiny-random-forests/device/zephyr/kernel/fatal.c:32
#1  0x1000fbbe in k_sys_fatal_error_handler (reason=reason@entry=0, 
    esf=esf@entry=0x20001f10 <z_interrupt_stacks+1992>)
    at /home/jon/projects/tiny-random-forests/device/zephyr/kernel/fatal.c:46
#2  0x1000d508 in z_fatal_error (reason=reason@entry=0, 
    esf=esf@entry=0x20001f10 <z_interrupt_stacks+1992>)
    at /home/jon/projects/tiny-random-forests/device/zephyr/kernel/fatal.c:122
#3  0x1000f35c in z_arm_fatal_error (reason=reason@entry=0, 
    esf=esf@entry=0x20001f10 <z_interrupt_stacks+1992>)
    at /home/jon/projects/tiny-random-forests/device/zephyr/arch/arm/core/fatal.c:73
#4  0x1000c1ae in z_arm_fault (msp=<optimized out>, psp=<optimized out>, 
    exc_return=<optimized out>, callee_regs=<optimized out>)
    at /home/jon/projects/tiny-random-forests/device/zephyr/arch/arm/core/cortex_m/fault.c:1157
#5  0x1000c1e4 in z_arm_hard_fault ()
    at /home/jon/projects/tiny-random-forests/device/zephyr/arch/arm/core/cortex_m/fault_s.S:102
#6  <signal handler called>
#7  0x00000000 in ?? ()
Backtrace stopped: previous frame identical to this frame (corrupt stack?)
(gdb) 


## Answered


#### What range of models are feasible at all

model size total. `<< 1MB`, preferably 10-100 kB

Plot model size estimates for datasets.
With majority voting, best case scenario.

median around 10kB.
Majority under 100kB. OK.


## Out-of-scope

To keep the paper focused. Candidates for "future work".



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
