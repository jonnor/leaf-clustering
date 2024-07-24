
## Optimizing Random Forest-Based Inference on RISC-V MCUs at the Extreme Edge

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
