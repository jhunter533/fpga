#ifndef LIF_H
#define LIF_H

#include <ap_fixed.h>
#include <hls_math.h>

#define INPUTSIZE 3
#define HIDDENDIM1 64
#define HIDDENDIM2 32
#define OUTPUTSIZE 1
#define TIMESTEPS 16

#define SIGMOIDSCALE 4.0
#define SIGMOIDSHIFT 2.0
#define MINLOG -20.0
#define MAXLOG 2.0
typedef ap_fixed<16, 8> fixed_t;

void lifNeuron(fixed_t inputs[], fixed_t weights[], fixed_t &membranePotential,
               fixed_t &spikeOut, fixed_t leak, fixed_t spikeThreshold);

void outputNeuron(fixed_t inputs[], fixed_t weights[], fixed_t &output,
                  fixed_t leak);

fixed_t sigmoidSurrogate(fixed_t x);
fixed_t clampLog(fixed_t value);
#endif // !LIF_H
