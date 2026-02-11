#include "snn_accel.h"
#include "hls_math.h"
#include <cstring>

// Neuron state (optimized precision)
static neuron_t lif1_v[HIDDEN1_SIZE];
static neuron_t lif2_v[HIDDEN2_SIZE];
static neuron_t mu_v[NUM_ACTIONS];
static neuron_t logstd_v[NUM_ACTIONS];

// Weights (BRAM-optimized layout)
static weight_t fc1_w[HIDDEN1_SIZE][NUM_STATES];
static weight_t fc2_w[HIDDEN2_SIZE][HIDDEN1_SIZE];
static weight_t mu_w[NUM_ACTIONS][HIDDEN2_SIZE];
static weight_t logstd_w[NUM_ACTIONS][HIDDEN2_SIZE];

static weight_t fc1_b[HIDDEN1_SIZE];
static weight_t fc2_b[HIDDEN2_SIZE];
static weight_t mu_b[NUM_ACTIONS];
static weight_t logstd_b[NUM_ACTIONS];

// Gradients (higher precision for stability)
static accum_t fc1_g[HIDDEN1_SIZE][NUM_STATES];
static accum_t fc2_g[HIDDEN2_SIZE][HIDDEN1_SIZE];
static accum_t mu_g[NUM_ACTIONS][HIDDEN2_SIZE];
static accum_t logstd_g[NUM_ACTIONS][HIDDEN2_SIZE];

static accum_t fc1_bg[HIDDEN1_SIZE];
static accum_t fc2_bg[HIDDEN2_SIZE];
static accum_t mu_bg[NUM_ACTIONS];
static accum_t logstd_bg[NUM_ACTIONS];

// Trace buffers (partitioned for parallel access)
static neuron_t lif1_trace[TIME_STEPS][HIDDEN1_SIZE];
static neuron_t lif2_trace[TIME_STEPS][HIDDEN2_SIZE];

// Constants
const neuron_t V_THRESHOLD = 1.0;
const neuron_t V_RESET = 0.0;
const neuron_t DECAY = 0.9512;
const accum_t LEARNING_RATE = 0.0003;

static dtype stored_action, stored_logpi;
static dtype stored_state[NUM_STATES];

// Memory optimization pragmas
#pragma HLS BIND_STORAGE variable = fc1_w type = RAM_2P impl = BRAM
#pragma HLS BIND_STORAGE variable = fc2_w type = RAM_2P impl = BRAM
#pragma HLS BIND_STORAGE variable = mu_w type = RAM_2P impl = BRAM
#pragma HLS BIND_STORAGE variable = lif1_trace type = RAM_T2P impl = BRAM
#pragma HLS BIND_STORAGE variable = lif2_trace type = RAM_T2P impl = BRAM

// Array partitioning for parallel access
#pragma HLS ARRAY_PARTITION variable = fc1_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = fc2_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = mu_w complete dim = 1

// fast math functions
// check correctness later
dtype fast_tanh(dtype x) {
#pragma HLS INLINE
  if (x > 3.0)
    return 1.0;
  if (x < -3.0)
    return -1.0;
  dtype x2 = x * x;
  return x * (27.0 + x2) / (27.0 + 9.0 * x2);
}

accum_t fast_exp_neg(accum_t x) {
#pragma HLS INLINE
  if (x < -10.0)
    return 4.54e-5;
  if (x > 0.0)
    return 1.0;

  accum_t term = 1.0;
  accum_t result = 1.0;
  for (int i = 1; i <= 8; i++) {
#pragma HLS UNROLL
    term = term * x / i;
    result = result + term;
  }
  if (result < 0.0)
    return 0.0;
  if (result > 1.0)
    return 1.0;
  return result;
}

accum_t surrogateGradient(neuron_t v, neuron_t threshold) {
#pragma HLS INLINE
  accum_t ax = hls::fabs(v - threshold);
  if (ax > 8.0)
    return 0.0;
  return 0.5 * hls::exp(-ax);
}

void updateLIF(neuron_t &v_mem, accum_t input, neuron_t &voltage_trace) {
#pragma HLS INLINE
  voltage_trace = v_mem * DECAY + input;
  if (voltage_trace >= V_THRESHOLD) {
    v_mem = V_RESET;
  } else {
    v_mem = voltage_trace;
  }
}

void updateLI(neuron_t &v_mem, accum_t input, neuron_t &out) {
#pragma HLS INLINE
  v_mem = v_mem * DECAY + input;
  out = v_mem;
}

void initLIF() {
#pragma HLS INLINE
  for (int i = 0; i < HIDDEN1_SIZE; i++) {
#pragma HLS UNROLL
    lif1_v[i] = 0;
  }
  for (int i = 0; i < HIDDEN2_SIZE; i++) {
#pragma HLS UNROLL
    lif2_v[i] = 0;
  }
}

void initLI() {
#pragma HLS INLINE
  for (int i = 0; i < NUM_ACTIONS; i++) {
#pragma HLS UNROLL
    mu_v[i] = 0;
    logstd_v[i] = 0;
  }
}

void multistep(const dtype state_in[NUM_STATES]) {
#pragma HLS PIPELINE II = 1
  accum_t hidden1[HIDDEN1_SIZE];
  accum_t hidden2[HIDDEN2_SIZE];

  for (int t = 0; t < TIME_STEPS; t++) {
    // Layer 1: Parallel LIF processing
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
#pragma HLS UNROLL
      accum_t sum = fc1_b[i];
      for (int j = 0; j < NUM_STATES; j++) {
#pragma HLS UNROLL
        sum += fc1_w[i][j] * state_in[j];
      }
      updateLIF(lif1_v[i], sum, lif1_trace[t][i]);
    }

    // Layer 2: Parallel LIF processing
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
#pragma HLS UNROLL
      accum_t sum = fc2_b[i];
      for (int j = 0; j < HIDDEN1_SIZE; j++) {
#pragma HLS UNROLL
        sum += fc2_w[i][j] * lif1_trace[t][j];
      }
      updateLIF(lif2_v[i], sum, lif2_trace[t][i]);
    }
  }
}

void forwardprop(const dtype state_in[NUM_STATES], dtype *action_out,
                 dtype *logpi_out) {
#pragma HLS PIPELINE II = 1
  initLIF();
  initLI();
  multistep(state_in);

  // Output layers (parallel processing)
  neuron_t mu_val[NUM_ACTIONS];
  neuron_t logstd_val[NUM_ACTIONS];

  for (int i = 0; i < NUM_ACTIONS; i++) {
#pragma HLS UNROLL
    accum_t mu_sum = mu_b[i];
    accum_t logstd_sum = logstd_b[i];
    for (int j = 0; j < HIDDEN2_SIZE; j++) {
#pragma HLS UNROLL
      mu_sum += mu_w[i][j] * lif2_trace[TIME_STEPS - 1][j];
      logstd_sum += logstd_w[i][j] * lif2_trace[TIME_STEPS - 1][j];
    }
    updateLI(mu_v[i], mu_sum, mu_val[i]);
    updateLI(logstd_v[i], logstd_sum, logstd_val[i]);
  }

  // Compute corrected log_pi
  dtype u = mu_val[0];
  dtype tanh_u = fast_tanh(u);
  dtype tanh2 = tanh_u * tanh_u;
  dtype log_sigma = logstd_val[0];
  dtype correction = -hls::log(1.0 - tanh2 + 1e-6) + 1.386294361;
  dtype log_pi = -log_sigma + correction;

  if (log_pi > 2.0)
    log_pi = 2.0;
  else if (log_pi < -20.0)
    log_pi = -20.0;

  *action_out = tanh_u;
  *logpi_out = log_pi;
  stored_action = tanh_u;
  stored_logpi = log_pi;
}

void backprop(dtype grad_da, dtype grad_dlogp,
              const dtype state_in[NUM_STATES]) {
#pragma HLS PIPELINE II = 1

  // Zero gradients (parallel)
  for (int i = 0; i < HIDDEN1_SIZE; i++) {
#pragma HLS UNROLL
    for (int j = 0; j < NUM_STATES; j++) {
      fc1_g[i][j] = 0;
    }
    fc1_bg[i] = 0;
  }
  for (int i = 0; i < HIDDEN2_SIZE; i++) {
#pragma HLS UNROLL
    for (int j = 0; j < HIDDEN1_SIZE; j++) {
      fc2_g[i][j] = 0;
    }
    fc2_bg[i] = 0;
  }
  for (int i = 0; i < NUM_ACTIONS; i++) {
#pragma HLS UNROLL
    for (int j = 0; j < HIDDEN2_SIZE; j++) {
      mu_g[i][j] = 0;
      logstd_g[i][j] = 0;
    }
    mu_bg[i] = 0;
    logstd_bg[i] = 0;
  }

  // Compute output gradients
  dtype tanh_deriv = 1.0 - stored_action * stored_action;
  accum_t dL_dmu = grad_dlogp * (2.0 * stored_action) + grad_da * tanh_deriv;
  accum_t dL_dlogstd = grad_dlogp * (-1.0);

  // Backprop to hidden2 (parallel)
  accum_t grad_h2[HIDDEN2_SIZE];
  for (int j = 0; j < HIDDEN2_SIZE; j++) {
#pragma HLS UNROLL
    grad_h2[j] = dL_dmu * mu_w[0][j] + dL_dlogstd * logstd_w[0][j];
    mu_g[0][j] = dL_dmu * lif2_trace[TIME_STEPS - 1][j];
    logstd_g[0][j] = dL_dlogstd * lif2_trace[TIME_STEPS - 1][j];
  }
  mu_bg[0] = dL_dmu;
  logstd_bg[0] = dL_dlogstd;

  // Backprop through time (LIF2)
  for (int t = TIME_STEPS - 1; t >= 0; t--) {
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
#pragma HLS UNROLL
      accum_t sg = surrogateGradient(lif2_trace[t][i], V_THRESHOLD);
      accum_t total_grad = grad_h2[i] * sg;
      for (int j = 0; j < HIDDEN1_SIZE; j++) {
#pragma HLS UNROLL
        fc2_g[i][j] += total_grad * lif1_trace[t][j];
      }
      fc2_bg[i] += total_grad;
    }
  }

  // Backprop to hidden1 (parallel)
  accum_t grad_h1[HIDDEN1_SIZE];
  for (int j = 0; j < HIDDEN1_SIZE; j++) {
#pragma HLS UNROLL
    grad_h1[j] = 0;
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
      grad_h1[j] += grad_h2[i] * fc2_w[i][j];
    }
  }

  // Backprop through time (LIF1)
  for (int t = TIME_STEPS - 1; t >= 0; t--) {
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
#pragma HLS UNROLL
      accum_t sg = surrogateGradient(lif1_trace[t][i], V_THRESHOLD);
      accum_t total_grad = grad_h1[i] * sg;
      for (int j = 0; j < NUM_STATES; j++) {
#pragma HLS UNROLL
        fc1_g[i][j] += total_grad * state_in[j];
      }
      fc1_bg[i] += total_grad;
    }
  }

  // Apply SGD updates (parallel)
  for (int i = 0; i < HIDDEN1_SIZE; i++) {
#pragma HLS UNROLL
    for (int j = 0; j < NUM_STATES; j++) {
      fc1_w[i][j] -= LEARNING_RATE * fc1_g[i][j];
    }
    fc1_b[i] -= LEARNING_RATE * fc1_bg[i];
  }
  for (int i = 0; i < HIDDEN2_SIZE; i++) {
#pragma HLS UNROLL
    for (int j = 0; j < HIDDEN1_SIZE; j++) {
      fc2_w[i][j] -= LEARNING_RATE * fc2_g[i][j];
    }
    fc2_b[i] -= LEARNING_RATE * fc2_bg[i];
  }
  for (int i = 0; i < NUM_ACTIONS; i++) {
#pragma HLS UNROLL
    for (int j = 0; j < HIDDEN2_SIZE; j++) {
      mu_w[i][j] -= LEARNING_RATE * mu_g[i][j];
      logstd_w[i][j] -= LEARNING_RATE * logstd_g[i][j];
    }
    mu_b[i] -= LEARNING_RATE * mu_bg[i];
    logstd_b[i] -= LEARNING_RATE * logstd_bg[i];
  }
}

extern "C" {
void snnAccel(volatile bool *start_fwd, volatile bool *start_bwd,
              volatile bool *busy, volatile dtype state_in[NUM_STATES],
              volatile dtype grad_da_in[NUM_ACTIONS],
              volatile dtype grad_dlogp_in, volatile dtype *action_out,
              volatile dtype *logpi_out) {
#pragma HLS INTERFACE s_axilite port = start_fwd bundle = ctrl
#pragma HLS INTERFACE s_axilite port = start_bwd bundle = ctrl
#pragma HLS INTERFACE s_axilite port = busy bundle = ctrl
#pragma HLS INTERFACE s_axilite port = state_in bundle = ctrl
#pragma HLS INTERFACE s_axilite port = grad_da_in bundle = ctrl
#pragma HLS INTERFACE s_axilite port = grad_dlogp_in bundle = ctrl
#pragma HLS INTERFACE s_axilite port = action_out bundle = ctrl
#pragma HLS INTERFACE s_axilite port = logpi_out bundle = ctrl
#pragma HLS INTERFACE s_axilite port = return bundle = ctrl

  static bool weights_initialized = false;
  if (!weights_initialized) {
    // Initialize weights (deterministic pattern)
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
      for (int j = 0; j < NUM_STATES; j++) {
        fc1_w[i][j] = ((i + j) % 100 - 50) * 0.004;
      }
      fc1_b[i] = 0;
    }
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
      for (int j = 0; j < HIDDEN1_SIZE; j++) {
        fc2_w[i][j] = ((i + j) % 100 - 50) * 0.004;
      }
      fc2_b[i] = 0;
    }
    for (int i = 0; i < NUM_ACTIONS; i++) {
      for (int j = 0; j < HIDDEN2_SIZE; j++) {
        mu_w[i][j] = ((i + j) % 100 - 50) * 0.002;
        logstd_w[i][j] = ((i + j) % 100 - 50) * 0.002;
      }
      mu_b[i] = 0;
      logstd_b[i] = 0;
    }
    weights_initialized = true;
  }

  *busy = 1;

  if (*start_fwd) {
    for (int i = 0; i < NUM_STATES; i++) {
      stored_state[i] = state_in[i];
    }
    forwardprop(state_in, action_out, logpi_out);
    *start_fwd = 0;
  }

  if (*start_bwd) {
    backprop(grad_da_in[0], grad_dlogp_in, stored_state);
    *start_bwd = 0;
  }

  *busy = 0;
}
}
