
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "model_none.h"
#include "model_m2cgen.h"

#include "testdata.h"

// model_predict() should be defined by one of the includes


int run_tests() {

    const int samples = testdata_samples;
    const int features_length = testdata_features;

    const int repeats = 2;

    const long int time = 0;

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

    printf("test-complete samples=%d time=%d errors=%d \n",
        tests, time, errors
    );
}


int main()
{
    return run_tests();
}
