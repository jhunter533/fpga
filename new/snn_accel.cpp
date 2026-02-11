#include "snn_accel.h"
#include "ap_fixed.h"
#include "hls_math.h"
#include <cstring>

typdef ap_fixed<16, 8> dtype;

// Neuron state
static dtype lif1_v[HIDDEN1_SIZE];
static dtype lif2_v[HIDDEN2_SIZE];
static dtype mu_v[NUM_ACTIONS];
static dtype logstd_v[NUM_ACTIONS];

// Weights (stored in BRAM)
static dtype fc1_w[HIDDEN1_SIZE][NUM_STATES];
static dtype fc2_w[HIDDEN2_SIZE][HIDDEN1_SIZE];
static dtype mu_w[NUM_ACTIONS][HIDDEN2_SIZE];
static dtype logstd_w[NUM_ACTIONS][HIDDEN2_SIZE];

static dtype fc1_b[HIDDEN1_SIZE];
static dtype fc2_b[HIDDEN2_SIZE];
static dtype mu_b[NUM_ACTIONS];
static dtype logstd_b[NUM_ACTIONS];

// Gradients (accumulated during backward)
static dtype fc1_g[HIDDEN1_SIZE][NUM_STATES];
static dtype fc2_g[HIDDEN2_SIZE][HIDDEN1_SIZE];
static dtype mu_g[NUM_ACTIONS][HIDDEN2_SIZE];
static dtype logstd_g[NUM_ACTIONS][HIDDEN2_SIZE];

static dtype fc1_bg[HIDDEN1_SIZE];
static dtype fc2_bg[HIDDEN2_SIZE];
static dtype mu_bg[NUM_ACTIONS];
static dtype logstd_bg[NUM_ACTIONS];

// Trace buffers for BPTT
static dtype lif1_trace[TIME_STEPS][HIDDEN1_SIZE];
static dtype lif2_trace[TIME_STEPS][HIDDEN2_SIZE];

dtype fast_tanh() {}
dtype fast_exp_neg() {}
dtype surrogate_gradient() {}
void update_lif() {}
void update_li() {}
void init_neurons() {}
void backprop() {}
void forwardprop() {}
void multistep() {}
void snn_accel() {}
