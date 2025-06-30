#include "LIF.h"

fixed_t sigmoidSurrogate(fixed_t x) {
#pragma HLS INLINE
  fixed_t xScaled = SIGMOIDSCALE * 4.0;
  fixed_t expVal = hls::exp(-xScaled);
  return (1.0) / (1.0 + expVal);
}
void lifNeuron(fixed_t inputs[], fixed_t weights[], fixed_t &membranePotential,
               bool &spikeOut, fixed_t leak, fixed_t spikeThreshold) {
#pragma HLS PIPELINE II = 1
#pragma HLS ARRAY_PARTITION variable = inputs cyclic factor = 16
#pragma HLS ARRAY_PARTITION variable = weights cyclic factor = 16

  fixed_t weightsum = 0;

MAC_LOOP:
  for (int i = 0; i < INPUTSIZE; i++) {
#pragma HLS UNROLL factor = 4
    weightsum += inputs[i] * weights[i];
  }
  membranePotential = (membranePotential * leak) + weightsum;
  spikeOut = (membranePotential >= spikeThreshold) ? 1 : 0;
  if (spikeOut)
    membranePotential = 0;
}

void outputNeuron(fixed_t inputs[], fixed_t weights[], fixed_t &output,
                  fixed_t leak) {
#pragma HLS PIPELINE II = 1
#pragma HLS ARRAY_PARTITION variable = inputs cyclic factor = 16
#pragma HLS ARRAY_PARTITION variable = weights cyclic factor = 16

  fixed_t weightedSum = 0;

MAC_LOOP:
  for (int i = 0; i < HIDDENDIM2; i++) {
#pragma HLS UNROLL factor = 4
    weightedSum += inputs[i] * weights[i];
  }
  output = output * leak + weightedSum;
}
