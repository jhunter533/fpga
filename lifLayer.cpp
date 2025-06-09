#include "lifLayer.h"

void lifLayer(
  fixed_t inputs[INPUTSIZE],
  fixed_t weights[NEURONCOUNT][INPUTSIZE],
  fixed_t membranePotentials[NEURONCOUNT],
  bool outputSpikes[NEURONCOUNT],
  fixed_t leak=2.0,
  fixed_t spikeThreshold=THRESHOLD
){
  #pragma HLS ARRAY_PARTITION variable=weights complete dim=1
  #pragma HLS ARRAY_PARTITION variable=membranePotentials complete
  #pragma HLS ARRAY_PARTITION variable=outputSpikes complete

LayerLoop:
  for(int i=0;i<NEURONCOUNT;i++){
    #pragma HLS UNROLL
    lifNeuron(inputs,weights[i],membranePotentials[i],outputSpikes[i],leak,spikeThreshold);
  }
}
