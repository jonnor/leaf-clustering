
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
2021.
Francesco Daghero et. al

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



# Hardware acceleration

#### ST Machine Learning Core

Built into ST IMUs. Those with part numbers ending in -X.
Decision tree classifier. With feature extraction.
No support for multiple decision trees/ensembles.
https://www.st.com/content/st_com/en/ecosystems/MEMS-Sensors-Ecosystem-for-Machine-Learning.html
https://github.com/STMicroelectronics/STMems_Machine_Learning_Core

#### An Ultra-Low Energy Human Activity Recognition Accelerator for Wearable Health Applications
https://dl.acm.org/doi/10.1145/3358175

? how is classificatio done
