#ifndef LIF_LAYER_H
#define LIF_LAYER_H

#include "LIF.h"

void lifLayer(fixed_t inputs[INPUTSIZE],fixed_t weights[NEURONCOUNT][INPUTSIZE],fixed_t membranePotentials[NEURONCOUNT],bool outputSpikes[NEURONCOUNT],fixed_t leak=2.,fixed_t spikeThreshold=THRESHOLD);

#endif // !LIF_LAYER_H
