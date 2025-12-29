
## Optimizing Random Forest-Based Inference on RISC-V MCUs at the Extreme Edge
2022.
https://dl.acm.org/doi/10.1109/TCAD.2022.3199903
Enrico Tabanelli, Giuseppe Tagliavini, Luca Benini


Defines multiple inference strategies for decision trees / forests.
Uses floating point features, 32 bits / 1 word.

- Naive. Code generation of the entire if-else trees. 24 bytes per node
- DT-Loop. Using an array-of-structs. With 16 bytes per node. Leaf nodes are separate.
- DT-Rec. Using an array-of-struct. With 24 bytes per node. Integrated leaf node in decision node, so fewer nodes total.
- DT-Arr. Using separate arrays for: feature index, threshold, children (2x). 16 bytes per node.

Shows pseudo-code for the algorihtm, data structures.
Also shows disassembly of the inner loops.

Ran experiments on PULPissimo.
RISC-V mcu. But does not have data caches.
Tested on two datasets, Vehicle and MFeat (handwritten digits).

4.8× speedup, and 45% storage reduction over naive.
Found DT-Rec and DT-Arr to be pareto optimal over DT-Loop and Naive.


## Ultra-compact Binary Neural Networks for Human Activity Recognition on RISC-V Processors
https://dl.acm.org/doi/abs/10.1145/3457388.3458656
https://arxiv.org/abs/2205.12781v1
2021.
Francesco Daghero et. al

Code public at https://github.com/francescodaghero/ultracompactBNN for Pulpissimo RISC-V

Compared RF and Binary Neural Network.
On UniMiB-SHAR and a proprietary dataset.
Has an MCU optimized RF implementation, which they compare against.

> The LEAVES array stores the output prediction for all leaf nodes.
> Specifically, our implementation stores class probabilities for all classes in the leaves, rather than a single class label,
> as this yields higher accuracy
> and also matches the scikit-learn implementation, making the conversion of models easier

> To save memory, only the right child is stored,
> whereas the left child is implicitly equal to the following node in the array.

> Thresholds and probabilities are quantized to 8bit integer,
> since RI5CY does not have a FPU.
> indexes in the different arrays also use 8 or 16bit, depending on the maximum RF size,
to minimize the memory footprint.


> For RFs, we varied the number of trees and their maximum depth
> ...
> Interestingly, RF raw outperforms RF features, probably because the simple magnitude features proposed in [22] are not very informative for the model.
> 
> In terms of cycles, RFs clearly outperform BNNs for accuracy values < 60%, requiring around 2.5k cycles even for large models.
> However, we could not find a RF-based solution able to produce higher accuracy, without exceeding the entire 520kB of memory of the target HW.

? Should be possible to reach much higher than 59% with RF on UniMiB-SHAR ?
BNN reached 70% accuracy.
Some of the limitation is due to model size. Which our work would improve on.
How much?
But may also need more powerful feature engineering.

"Adaptive Random Forests for Energy-Efficient Inference on Microcontrollers" (by same authors)
also tests on UniMiB. Also struggled to RF above 60%.

"High-Level Features for Human Activity Recognition and Modeling" reached up to 67.3% accuracy on UniMiB SHAR.

Thesis: Most of performance gain of BNN over RF comes from the convolutional layer

> our results of Section 4 show that, for HAR, a good accuracy can be obtained with 1D BNNs including as little as 2 or 4 channels

For Walk dataset 2x layers with 2 channels, 7 kernel length outperformed RF features.
For UniMiB, needed 3 layers and additional pooling. Conv(8,15),Conv(32,7),Pool(4,4),Conv(32,7).
RF uses 1-2 orders of magnitude less CPU cycles.

? dilated should have helped with the larger receptive field for UniMiB.
Maybe even downsampling.


#### New machine learning approaches for real-life human activity recognition using smartphone sensor-based data
https://www.sciencedirect.com/science/article/pii/S0950705123000102

Daniel Garcia-Gonzalez, et al.
University of A Coruña.
2023.

- comparison of the main machine learning algorithms for HAR.
- A dataset taken in a real-life environment was used, unlike in other studies.
- Experiments were done to get the best model configurations for long-themed activities.

> These experiments proved that, in this case, tree-based models, such as Random Forest, outperform the rest.
> The final result shows an enormous improvement in the accuracy of the best model found to date for this purpose, from 74.39% to 92.97%.

Custom dataset. Published to https://lbd.udc.es/research/real-life-HAR-dataset/
Classes. Inactive, active, walking, driving
Four different sensors: accelerometer, gyroscope, magnetometer and GPS/
Smartphone collection. 19 individuals.
Not fixed orientation or placement of the smartphone,



#### Tree Model Quantization for Embedded Machine Learning Applications

Summarization of talk at TinyML 2021
tinyML Summit 2021 Partner Session: Tree Ensemble Model Compression for Embedded Machine Learning
https://www.youtube.com/watch?v=KTjkWSU6R2o
Leslie Schradin

https://sensei.tdk.com/tree-model-quantization-for-embedded-machine-learning-applications/

> In our experience at Qeexo we have found that tree-based models often outperform
> all other model types when the models are constrained to be small in size (e.g. < 100s KB)
> or when there is relatively little available data (e.g. < 100,000 training examples).
> Both of these conditions are often true for machine learning problems targeting embedded devices.

For Qeexo AutoML.
Leaf quantization using affine transformation. Scaling and shift.
Across all leaf values in a single tree.
Used 8 bit leaves for the example.

Also showed 8 bit feature tresholds, but have to dequantize and store per-feature transforms. Net gain low.
In their example, leaves and nodes took of original float model took approx same space.

Parent. Abandoned as of December 2025
https://patents.google.com/patent/US20220114457A1/en

#### Fast Inference of Tree Ensembles on ARM Devices
https://arxiv.org/abs/2305.08579
May 2023
Simon Koschel, Sebastian Buschjäger, Claudio Lucchese, Katharina Morik

QuickScorer/RAPIDSCORER are SIMD evaluations for decision tree ensembles.

> In this paper, we convert the popular QuickScorer algorithm and its siblings from Intel's AVX to ARM's NEON instruction set.
> Second, we extend our implementation from ranking models to classification models such as Random Forests.
> Third, we investigate the effects of using fixed-point quantization in Random Forests.
> Our study shows that a careful implementation of tree traversal on ARM CPUs leads to a speed-up of up to 9.4 compared to a reference implementation


Compared 16 bit integers and 32 bit floatsfor leaf nodes and splits.
On 4 dataset int16 gave same performance.
But on EEG dataset, quantization of the leaf values leads to a drop of nearly 4 percentage points.

#### Boosted Trees on a Diet: Compact Models for Resource-Constrained Devices
https://arxiv.org/abs/2510.26557
Jan Stenkamp, Nina Herrmann, Benjamin Karic, Stefan Oehmcke, Fabian Gieseke

Training compact boosted decision tree ensembles that exhibit a reduced memory footprint by rewarding,
among other things, the reuse of features and thresholds during training.
Our experimental evaluation shows that models achieved
the same performance with a compression ratio of 4-16x compared to LightGBM models.

#### Joint leaf-refinement and ensemble pruning through regularization
Published: 15 March 2023
https://link.springer.com/article/10.1007/s10618-023-00921-z

Sebastian Buschjäger & Katharina Morik.
TU Dortmund University

> leaf-refinement is a technique that improves the performance of a tree ensemble
> by jointly re-learning the probability estimates in the leaf nodes of the trees
> introduce L1 regularization into the leaf-refinement objective,
> which allows us to jointly prune and refine trees at the same time

Stores probabilities in each leaf node can be stored within a 2 Byte (int16).

> This operation is also implemented in FastInference,
> and we could not detect any change in the accuracy with this quantization.

Deployed a 20 kB model to PhyNode (MSP430 MCU).

https://github.com/sbuschjaeger/fastinference?tab=readme-ov-file

#### Compressing tree ensembles through Level-wise Optimization and Pruning
https://proceedings.mlr.press/v267/devos25a.html

ICML 2025.

Presents LOP, a method for compressing a given tree ensemble by pruning or entirely removing trees in it,
while updating leaf predictions in such a way that predictive accuracy is mostly unaffected. 

Tested both Random Forest and XGB models.
For RF, showed extreme compressions of 100-10000x.

! shows connection to original model Pareto fronts

Compares with Global Refinement.

#### Global refinement of random forest
S Ren, X Cao, Y Wei, J Sun
Proceedings of the IEEE conference on computer vision and 2015•
https://openaccess.thecvf.com/content_cvpr_2015/papers/Ren_Global_Refinement_of_2015_CVPR_paper.pdf

Describes how to refine leaf values across an entire forest as a post-processing step.

#### ResOT: Resource-Efficient Oblique Trees for Neural Signal Classification

Bingzhao Zhu, Masoud Farivar, and Mahsa Shoaran
2020
IEEE TRANSACTIONS ON BIOMEDICAL CIRCUITS AND SYSTEMS

As opposed to axis-aligned decision trees, oblique trees use multiple features to make splits.

> Training oblique trees is not a trivial task, as the weights are not differentiable. In addition, without compressing the tree structure, oblique trees may grow overly complex and use many features to make splits, increasing both the model size and node complexity.
> To tackle these issues, we introduce the oblique trees with probabilistic splits.
> Rather than deterministically routing a decision tree, we send a sample to the left or right subtree based on a probability value.
> With this probabilistic routing, we can derive the objective function and train oblique trees with gradient-based optimization algorithms, and use various compression techniques applied to neural networks.



# Ideas

### Positional encoding in demand trees

IIR or FIR filters (convolution) can be used.
Doing a single level of adaptive filters is already quite powerful.
Traditional non-neural / non-backprop algorithms like RF limited because this is not immediately available.
The kernels could be selected randomly (see Rocket).
Or use some standard basis functions or filters.

Would want to aggregate the outputs of such a kernel,
to make it accessible to an RF classifier.
- mean should allow to determine
- max should allow to determine whether a given subsequence was present (at any location).
- adding the kernel sum at all points as deterministic feature indices, should allow a concept of location
But shift invariance is hard to do.

For many filters/kernel running with step=1 might give very correlated outputs, not particularly informative.
Other kernels such as an edge detector may need to run on each step.
Would save a lot of computation by going step 2,4,8 etc.
Dilation might be a relevant concept here.
Ideally the step size would be matched to the kernel type.
Sample rate adaption also highly releated.

# Hyperparameter tuning

## Best first trees

best-first (leaf-wise) is an alternative tree growth strategy to the traditional depth-first (level-wise).

Best-first Decision Tree Learning (Thesis, Master of Science (MSc)).
Shi, H. (2007).
The University of Waikato, Hamilton, New Zealand.
Retrieved from https://hdl.handle.net/10289/2317

In theory produces better trees when the number of leaves are restricted.

https://datascience.stackexchange.com/questions/26699/decision-trees-leaf-wise-best-first-and-level-wise-tree-traverse

Thesis compared to minimal cost-complexity pruning.

This mode is used in scikit-learn when setting the `max_leaf_nodes` setting.
This setting should allow to quite predictably adjust the size of the resulting tree/forest?


## Hyperparameters and Tuning Strategies for Random Forest

Looks at 500++ trees.
References many other relevant works.


## Better Trees: An empirical study on hyperparameter tuning of classification decision tree induction algorithms

Does not consider ensembles. Only CART and J48.


### A time-efficient convolutional neural network model in human activity recognition
https://link.springer.com/article/10.1007/S11042-020-10435-1

> find that with removing the pooling layers and instead adding strides to convolution layers,
> the computational time will decrease notably while the model performance will not change or in some cases will even improve.

> impact of applying fast fourier transform (FFT) to inputs before training learning algorithm.
> It will be shown that this preprocessing will enhance the model performance.

### Deep Neural Networks for Sensor-Based Human Activity Recognition Using Selective Kernel Convolution
https://ieeexplore.ieee.org/abstract/document/9507456
2021.

> an attention idea to perform kernel selection among multiple branches with different RFs (receptive fields)
> it can achieve a higher recognition accuracy under a similar computing budget.

# Hardware acceleration

#### ST Machine Learning Core

Built into ST IMUs. Those with part numbers ending in -X.
Decision tree classifier. With feature extraction.
? No support for multiple decision trees/ensembles
https://www.st.com/content/st_com/en/ecosystems/MEMS-Sensors-Ecosystem-for-Machine-Learning.html
https://github.com/STMicroelectronics/STMems_Machine_Learning_Core

#### An Ultra-Low Energy Human Activity Recognition Accelerator for Wearable Health Applications
https://dl.acm.org/doi/10.1145/3358175

? how is classificatio done


# Less related

#### Human Activity Recognition on Microcontrollers with Quantized and Adaptive Deep Neural Networks
Focuses more on CNN. With RF as baseline

https://dl.acm.org/doi/10.1145/3542819
August 2022
Daghero et al.

Proposing a set of efficient one-dimensional Convolutional.
Experiments on four datasets. UniMiB-SHAR. UCI HAPT. WISDM. WALK.
Targeting an ultra-low-power RISC-V MCU.
Pareto-optimal CNNs for HAR, compared to Random Forest baseline.

Inference latency that ranges between 9 μs and 16 ms.
Their memory occupation varies in 0.05–23.17 kB.

Fixed-precision integer quantizations, namely, 8-bit, 4-bit, 2-bit, and 1-bit.
PArameterized Clipping acTivation (PACT) quantization algorithm.
Conv1d with Maxpool, followed by fully connected.
