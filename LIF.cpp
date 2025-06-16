#include "LIF.h"

void lifNeuron(
  fixed_t inputs[],
  fixed_t weights[],
  fixed_t &membranePotential,
  bool &spikeOut,
  fixed_t leak,
  fixed_t spikeThreshold
)
{
  #pragma HLS PIPELINE II=1
  #pragma HLS ARRAY_PARTITION variable=inputs complete
  #pragma HLS ARRAY_PARTITION variable=weights complete

  fixed_t weightsum=0;

  MAC_LOOP:
  for(int i=0;i<INPUTSIZE;i++){
    #pragma HLS UNROLL
    weightsum+=inputs[i]*weights[i];
  }
  membranePotential=(membranePotential*leak)+weightsum;
  spikeOut=(membranePotential>=spikeThreshold)?1:0;
  if(spikeOut)membranePotential=0;
}

void outputNeuron(
  fixed_t inputs[],
  fixed_t weights[],
  fixed_t &output,
  fixed_t leak
){
  #pragma HLS PIPELINE II=1
  #pragma HLS ARRAY_PARTITION variable=inputs complete
  #pragma HLS ARRAY_PARTITION variable=weights complete

  fixed_t weightedSum=0;

MAC_LOOP:
  for(int i=0;i<HIDDENDIM2;i++){
    #pragma HLS UNROLL
    weightedSum+=inputs[i]*weights[i];
  }
  output=output*leak+weightedSum;
}
