#ifndef SNN_ACCEL_H
#define SNN_ACCEL_H

#include "ap_fixed.h"
#include "hls_math.h"

const int NUM_STATES = 3;
const int HIDDEN1_SIZE = 64;
const int HIDDEN2_SIZE = 32;
const int NUM_ACTIONS = 1;
const int TIME_STEPS = 5;

typedef ap_fixed<16, 8> weight_t;
typedef ap_fixed<18, 10> accum_t;
typedef ap_fixed<12, 4> neuron_t;
typedef ap_fixed<16, 8> dtype;

#endif
