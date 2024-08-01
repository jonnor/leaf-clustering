
import pickle
import sys

import pandas

import emlearn
import m2cgen
import micromlgen

from emlearn.evaluate.size import get_program_size, check_build_tools

from src.experiments.metrics import unique_features, feature_counts

def export_emlearn(estimator, inference='loadable', dtype='int16_t', **kwargs):

    c = emlearn.convert(estimator, dtype=dtype, **kwargs)
    code = c.save(name='model', inference=inference)    

    return code

def export_m2cgen(estimator, **kwargs):

    code = m2cgen.export_to_c(estimator)

    return code

def export_micromlgen(estimator, **kwargs):

    code = micromlgen.port(estimator)

    return code


def generate_test_program(model_code, features):

    # XXX: the cast to float is wrong. Will crash horribly during execution
    # Only works for size estimation

    # FIXME: implement inference for all types. emlearn, m2cgen, micromlgen

    model_code += f"""
    int {model_name}_predict(const {dtype} *f, int l) {{
        return eml_trees_predict(&{model_name}, (float *)f, l);
    }}"""


    test_program = \
    f"""
    #include <stdint.h>

    #if {model_enabled}
    {model_code}

    static {dtype} features[{features_length}] = {{0, }};
    #endif

    int main()
    {{
        uint8_t pred = 0;
        #if {model_enabled}
        pred = {model_name}_predict(features, {features_length});
        #endif
        int out = pred;
        return out;
    }}
    """


def main():

    # Include test data. Features + expected label. Small set, say 10x per class
    # Inference code for each framework. One file per framework. Defines predict function
    # Test program. Macros to select a framework.
    #   Conditional include
    # Measure program size with no/dummy framework

    # One directory per model
    # Select N models, of various sizes, for each dataset

    estimator_path = sys.argv[1]

    platforms = pandas.DataFrame.from_records([
        ('arm', 'Cortex-M0'),
        ('arm', 'Cortex-M4F'),
    ], columns=['platform', 'cpu'])

    with open(estimator_path, 'rb') as f:
        estimator = pickle.load(f)

    m = estimator

    counts = feature_counts(m)
    multi = counts[counts > 1]
    print('features used more than once', len(multi))

    features = unique_features(m)
    print('unique features', features)

    ce = export_emlearn(m)
    c2 = export_m2cgen(m)
    cu = export_micromlgen(m)

    print(len(ce)/1000, len(c2)/1000, len(cu)/1000)

    data = get_program_size(test_program, platform=platform, mcu=mcu)

    return pandas.Series(data)



if __name__ == '__main__':
    main()
