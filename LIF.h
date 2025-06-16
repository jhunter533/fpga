#ifndef LIF_H
#define LIF_H

#include <ap_fixed.h>

#define INPUTSIZE 3
#define HIDDENDIM1 64
#define HIDDENDIM2 64
#define OUTPUTSIZE 1

typedef ap_fixed<16,8> fixed_t;

void lifNeuron(
  fixed_t inputs[],
  fixed_t weights[],
  fixed_t &membranePotential,
  bool &spikeOut,
  fixed_t leak,
  fixed_t spikeThreshold
);

void outputNeuron(
  fixed_t inputs[],
  fixed_t weights[],
  fixed_t &output,
  fixed_t leak
);
#endif // !LIF_H
