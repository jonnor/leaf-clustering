
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

// For eml_benchmark_micros(), used for timing
#include <eml_benchmark.h>

// Defines model_predict()
#if defined(MODEL_NONE)
    #include "model_none.h"
#elif defined(MODEL_M2CGEN)
    #include "model_m2cgen.h"
#elif defined(MODEL_EMLEARN)
    #include "model_emlearn.h"
#elif defined(MODEL_MICROMLGEN)
    #include "model_micromlgen.h"
#else
    #error "No model defined"
#endif

// Defines testdata_samples, _features, _values, _labels
#include "testdata.h"


int run_tests() {

    const int samples = testdata_samples;
    const int features_length = testdata_features;

    const int repeats = 1000;

    const int64_t start_time = eml_benchmark_micros();

    int errors = 0;
    int tests = 0;
    for (int i=0; i<samples; i++) {
        const int16_t *features = testdata_values + (i*features_length);
        const int expect = testdata_labels[i];

        for (int r=0; r<repeats; r++) {

            const int out = model_predict(features, features_length);

            const bool correct = (out == expect);
            if (!correct) {
                errors += 1;
            }
            tests += 1;
        }
    }
    errors = errors / repeats;
    tests = tests / repeats;
    const int64_t time_us = eml_benchmark_micros() - start_time;

    printf("test-complete samples=%d time=%ld errors=%d \n",
        tests, (long)time_us, errors
    );

    return errors;
}


int main()
{
    return run_tests();
}
