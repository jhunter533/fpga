#include "LIF.h"

void lifTop(
  hls::stream<fixed_t> &input_stream,
  hls::stream<fixed_t> &weight_stream,
  hls::stream<bool> &output_stream,
  fixed_t &membrane_state
){
  #pragma HLS INTERFACE axis port=input_stream
  #pragma HLS INTERFACE axis port=weight_stream
  #pragma HLS INTERFACE axis port=output_stream
  #pragma HLS INTERFACE ap_vld port=membrane_state
  #pragma HLS PIPELINE II=1

  static fixed_t inputs[CONNECTIONS];
  static fixed_t weights[CONNECTIONS];
  #pragma HLS ARRAY_PARTITION variable=inputs complete
  #pragma HLS ARRAY_PARTITION variable=weights complete

  static fixed_t membrane=0;
  bool spike;
  fixed_t leak, spikeThresh;

  for(int i=0;i<CONNECTIONS;i++){
    #pragma HLS UNROLL
    inputs[i]=input_stream.read();
    weights[i]=weight_stream.read();
  }
  lifNeuron(inputs,weights,membrane,spike,leak,spikeThresh);
  output_stream.write(spike);
  membrane_state=membrane;
}
