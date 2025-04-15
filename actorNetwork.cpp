#include "actorNetwork.hpp"

fixed_t relu(fixed_t x){
  return x>0?x:fixed_t(0);
}

void policyNetwork(
  hls::stream<fixed_t> &stateIn,
  hls::stream<fixed_t> &mu_out,
  hls::stream<fixed_t> &log_std_out,
  const NetworkParams &params,
){
  #pragma HLS INTERFACE axis port=state_in
    #pragma HLS INTERFACE axis port=mu_out
    #pragma HLS INTERFACE axis port=log_std_out
    #pragma HLS INTERFACE s_axilite port=params
    #pragma HLS INTERFACE ap_ctrl_none port=return

  // Layer 1
    fixed_t hidden1[HIDDEN_DIM];
    STATE_LOOP: for(int i = 0; i < HIDDEN_DIM; i++) {
        #pragma HLS PIPELINE II=1
        fixed_t sum = params.fc1_bias[i];
        for(int j = 0; j < STATE_DIM; j++) {
            sum += params.fc1_weights[i][j] * state_in.read();
        }
        hidden1[i] = relu(sum);
    }

    // Layer 2
    fixed_t hidden2[HIDDEN_DIM];
    HIDDEN_LOOP: for(int i = 0; i < HIDDEN_DIM; i++) {
        #pragma HLS PIPELINE II=1
        fixed_t sum = params.fc2_bias[i];
        for(int j = 0; j < HIDDEN_DIM; j++) {
            sum += params.fc2_weights[i][j] * hidden1[j];
        }
        hidden2[i] = relu(sum);
    }

    // Output layers
    fixed_t mu[ACTION_DIM], log_std[ACTION_DIM];
    OUTPUT_LOOP: for(int i = 0; i < ACTION_DIM; i++) {
        #pragma HLS UNROLL
        fixed_t mu_sum = 0;
        fixed_t log_std_sum = 0;
        
        for(int j = 0; j < HIDDEN_DIM; j++) {
            mu_sum += params.mu_weights[i][j] * hidden2[j];
            log_std_sum += params.log_std_weights[i][j] * hidden2[j];
        }
        
        mu[i] = mu_sum;
        log_std[i] = log_std_sum;
    }

    // Write outputs
    for(int i = 0; i < ACTION_DIM; i++) {
        mu_out.write(mu[i]);
        log_std_out.write(log_std[i]);
    }

}
