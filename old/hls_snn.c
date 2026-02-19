#include "ap_fixed.h"
#include "hls_stream.h"
#include <ap_int.h>

typedef ap_fixed<18, 2> fxp_input;
typedef ap_fixed<18, 2> fxp_weight;
typedef ap_fixed<24, 8> fxp_neuron;
typedef ap_fixed<16, 0> fxp_bias;

#define NUM_STATES 3
#define HIDDEN1_SIZE 64
#define HIDDEN2_SIZE 32
#define NUM_ACTIONS 1
#define TIME_STEPS 5
#define V_THRESHOLD 1024
#define TAU_MEMBRANe_FXP 20480

// BRAM weight storage
struct WeightMem {
  fxp_weight fc1_weights[HIDDEN1_SIZE][NUM_STATES];
  fxp_weight fc2_weights[HIDDEN2_SIZE][HIDDEN1_SIZE];
  fxp_weight mu_weights[NUM_ACTIONS][HIDDEN2_SIZE];
  fxp_weight logstd_weights[NUM_ACTIONS][HIDDEN2_SIZE];

  fxp_bias fc1_bias[HIDDEN1_SIZE];
  fxp_bias fc2_bias[HIDDEN2_SIZE];
  fxp_bias mu_bias[NUM_ACTIONS];
  fxp_bias logstd_bias[NUM_ACTIONS];
};

WeightMem weights_memory;

// Fixed-point tanh approximation using lookup table
fxp_neuron fast_tanh_fxp(fxp_neuron x) {
  // Simplified tanh approximation: x / (1 + |x|) for |x| < 3
  fxp_neuron abs_x = x > 0 ? x : -x;
  if (abs_x > 3)
    return x > 0 ? 1 : -1;
  return x / (1 + abs_x);
}

// LIF neuron update
fxp_neuron update_lif_neuron(fxp_neuron current_voltage,
                             fxp_neuron input_current) {
  // Exponential decay: v(t+1) = v(t) * exp(-dt/tau) + input
  fxp_neuron decay_factor = 983; // Fixed-point: 0.96 (approx exp(-1/20))
  fxp_neuron new_voltage =
      current_voltage * decay_factor / 1024 + input_current;

  if (new_voltage > V_THRESHOLD) {
    return 0; // Reset to 0 after spike
  }
  return new_voltage;
}

// LI neuron update (for continuous output)
fxp_neuron update_li_neuron(fxp_neuron current_voltage,
                            fxp_neuron input_current) {
  fxp_neuron decay_factor = 950; // Fixed-point: 0.93 (leak rate)
  return current_voltage * decay_factor / 1024 + input_current;
}

// Matrix multiplication with optimization
void matrix_multiply_fxp(const fxp_weight weights[][NUM_STATES],
                         const fxp_input input[NUM_STATES],
                         fxp_neuron output[HIDDEN1_SIZE], int rows) {
#pragma HLS INLINE
#pragma HLS PIPELINE II = 1

  for (int i = 0; i < rows; i++) {
    fxp_neuron sum = 0;
#pragma HLS UNROLL factor = 2
    for (int j = 0; j < NUM_STATES; j++) {
      sum += weights[i][j] * input[j];
    }
    output[i] = sum;
  }
}

void snn_hls_kernel(hls::stream<fxp_input> &state_in,
                    hls::stream<fxp_neuron> &action_out,
                    hls::stream<fxp_neuron> &logprob_out) {
  // HLS pragmas for interface
#pragma HLS INTERFACE axis port = state_in
#pragma HLS INTERFACE axis port = action_out
#pragma HLS INTERFACE axis port = logprob_out
#pragma HLS INTERFACE s_axilite port = return bundle = control

  // BRAM resource allocation for weights
#pragma HLS RESOURCE variable = weights_memory.fc1_weights core = RAM_2P_BRAM
#pragma HLS RESOURCE variable = weights_memory.fc2_weights core = RAM_2P_BRAM
#pragma HLS RESOURCE variable = weights_memory.mu_weights core = RAM_2P_BRAM
#pragma HLS RESOURCE variable = weights_memory.logstd_weights core = RAM_2P_BRAM

// Array partitioning for parallel access
#pragma HLS ARRAY_PARTITION variable =                                         \
    weights_memory.fc1_weights complete dim = 1
#pragma HLS ARRAY_PARTITION variable =                                         \
    weights_memory.fc2_weights complete dim = 1
#pragma HLS ARRAY_PARTITION variable =                                         \
    weights_memory.mu_weights complete dim = 1
#pragma HLS ARRAY_PARTITION variable =                                         \
    weights_memory.logstd_weights complete dim = 1

#pragma HLS ARRAY_PARTITION variable = weights_memory.fc1_bias complete
#pragma HLS ARRAY_PARTITION variable = weights_memory.fc2_bias complete
#pragma HLS ARRAY_PARTITION variable = weights_memory.mu_bias complete
#pragma HLS ARRAY_PARTITION variable = weights_memory.logstd_bias complete

  // Neuron state arrays
  fxp_neuron hidden1[HIDDEN1_SIZE];
  fxp_neuron hidden2[HIDDEN2_SIZE];
  fxp_neuron temp_hidden1[HIDDEN1_SIZE];
  fxp_neuron temp_hidden2[HIDDEN2_SIZE];

#pragma HLS ARRAY_PARTITION variable = hidden1 complete
#pragma HLS ARRAY_PARTITION variable = hidden2 complete
#pragma HLS ARRAY_PARTITION variable = temp_hidden1 complete
#pragma HLS ARRAY_PARTITION variable = temp_hidden2 complete

  // Read input state
  fxp_input state[NUM_STATES];
#pragma HLS ARRAY_PARTITION variable = state complete
  for (int i = 0; i < NUM_STATES; i++) {
    state[i] = state_in.read();
  }

  // Initialize neuron states
  for (int i = 0; i < HIDDEN1_SIZE; i++) {
    hidden1[i] = 0;
  }
  for (int i = 0; i < HIDDEN2_SIZE; i++) {
    hidden2[i] = 0;
  }

  // Time-stepped SNN simulation
  for (int t = 0; t < TIME_STEPS; t++) {
    // Layer 1: state -> hidden1
    matrix_multiply_fxp(weights_memory.fc1_weights, state, temp_hidden1,
                        HIDDEN1_SIZE);

#pragma HLS PIPELINE II = 1
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
      fxp_neuron input = temp_hidden1[i] + weights_memory.fc1_bias[i];
      hidden1[i] = update_lif_neuron(hidden1[i], input);
    }

// Layer 2: hidden1 -> hidden2
#pragma HLS PIPELINE II = 1
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
      fxp_neuron sum = 0;
#pragma HLS UNROLL factor = 4
      for (int j = 0; j < HIDDEN1_SIZE; j++) {
        sum += weights_memory.fc2_weights[i][j] * hidden1[j];
      }
      temp_hidden2[i] = sum + weights_memory.fc2_bias[i];
      hidden2[i] = update_lif_neuron(hidden2[i], temp_hidden2[i]);
    }
  }

  // Output layers: LI neurons for continuous output
  fxp_neuron mu_output[NUM_ACTIONS];
  fxp_neuron logstd_output[NUM_ACTIONS];

#pragma HLS ARRAY_PARTITION variable = mu_output complete
#pragma HLS ARRAY_PARTITION variable = logstd_output complete

  // Matrix multiply for mu (action output)
  for (int i = 0; i < NUM_ACTIONS; i++) {
    fxp_neuron sum = 0;
#pragma HLS UNROLL
    for (int j = 0; j < HIDDEN2_SIZE; j++) {
      sum += weights_memory.mu_weights[i][j] * hidden2[j];
    }
    mu_output[i] = update_li_neuron(sum, weights_memory.mu_bias[i]);
    fxp_neuron action = fast_tanh_fxp(mu_output[i]);
    action_out.write(action);
  }

  // Matrix multiply for logstd
  for (int i = 0; i < NUM_ACTIONS; i++) {
    fxp_neuron sum = 0;
#pragma HLS UNROLL
    for (int j = 0; j < HIDDEN2_SIZE; j++) {
      sum += weights_memory.logstd_weights[i][j] * hidden2[j];
    }
    logstd_output[i] = update_li_neuron(sum, weights_memory.logstd_bias[i]);
    logprob_out.write(logstd_output[i]);
  }
}

// Weight loading function (separate kernel or function)
void load_weights(fxp_weight *new_weights, fxp_bias *new_biases) {
#pragma HLS INTERFACE m_axi port = new_weights offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = new_biases offset = slave bundle = gmem1
#pragma HLS INTERFACE s_axilite port = return bundle = control

  // Load FC1 weights
  for (int i = 0; i < HIDDEN1_SIZE; i++) {
    for (int j = 0; j < NUM_STATES; j++) {
#pragma HLS PIPELINE II = 1
      weights_memory.fc1_weights[i][j] = new_weights[i * NUM_STATES + j];
    }
  }

  // Load FC2 weights
  for (int i = 0; i < HIDDEN2_SIZE; i++) {
    for (int j = 0; j < HIDDEN1_SIZE; j++) {
#pragma HLS PIPELINE II = 1
      weights_memory.fc2_weights[i][j] =
          new_weights[HIDDEN1_SIZE * NUM_STATES + i * HIDDEN1_SIZE + j];
    }
  }

  // Load bias weights similarly...
  // Load output weights...
}
