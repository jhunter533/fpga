#include "snnActor.h"

void actorNetwork(
  fixed_t state[INPUTSIZE],
  fixed_t &action,
  fixed_t w1,
  fixed_t w2,
  fixed_t w3,
  fixed_t leak,
  fixed_t threshold
){
  #pragma HLS INTERFACE ap_ctrl_none port=return
  #pragma HLS INTERFACE ap_fifo port=state
  #pragma HLS INTERFACE ap_fifo port=action

  #pragma HLS ARRAY_PARTITION variable=w1 complete dim=0
  #pragma HLS ARRAY_PARTITION variable=w2 complete dim=0
  #pragma HLS ARRAY_PARTITION variable=w3 complete dim=0

  static fixed_t mem1[HIDDENDIM1]={0};
  static fixed_t mem2[HIDDENDIM2]={0};
  static fixed_t memOut[OUTPUTSIZE]={0};
  #pragma HLS ARRAY_PARTITION variable=mem1 complete
  #pragma HLS ARRAY_PARTITION variable=mem2 complete
  #pragma HLS ARRAY_PARTITION variable=memOut complete

  bool spikes1[HIDDENDIM1]={0};
  bool spikes2[HIDDENDIM2]={0};

  fixed_t spikeIn2[HIDDENDIM1];
  fixed_t spikeIn3[HIDDENDIM2];
  #pragma HLS ARRAY_PARTITION variable=spikes1 complete
  #pragma HLS ARRAY_PARTITION variable=spikes2 complete
  #pragma HLS ARRAY_PARTITION variable=spikIn2 complete
  #pragma HLS ARRAY_PARTITION variable=spikeIn3 complete

LAYER1:
  for(int n=0;n<HIDDENDIM1;n++){
    #pragma HLS UNROLL
    lifNeuron(state,w1[n],mem1[n],spikes1[n],leak,threshold);
  }

SPIKE_CONV1:
  for(int i=0;i<HIDDENDIM1;i++){
    #pragma HLS UNROLL
    spikeIn2[i]=spikes1[i]?fixed_t(1.0):fixed_t(0.0);
  }

LAYER2:
  for(int n=0;n<HIDDENDIM2;n++){
    #pragma HLS UNROLL
    lifNeuron(spikeIn2,w2[n],mem2[n],spikes2[n],leak,threshold);
  }

SPIKE_CONV2:
  for(int i=0;i<HIDDENDIM2;i++){
    #pragma HLS UNROLL
    spikeIn3[i]=spikes2[i]?fixed_t(1.0):fixed_t(0.0);
  }

outputNeuron(spikeIn3,w3[0],memOut[0],leak);

action=4.0*memOut[0]-2.0;

}
