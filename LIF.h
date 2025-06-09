#ifndef LIF_H
#define LIF_H

#include <ap_fixed.h>

#define CONNECTIONS 16
#define THRESHOLD 1.0

typedef ap_fixed<16,8> fixed_t;

void lifNeuron(
  fixed_t inputs[CONNECTIONS],
  fixed_t weights[CONNECTIONS],
  fixed_t &membranePotential,
  bool &spikeOut,
  fixed_t leak=2.,
  fixed_t spikeThreshold=THRESHOLD
);

#include <hls_stream.h>
void lifTop(hls::stream<fixed_t> &input_stream, hls::stream<fixed_t> &weight_stream,hls::stream<bool> &output_stream,fixed_t &membrane_state);
#endif // !LIF_H
