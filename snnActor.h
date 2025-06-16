#ifndef SNN_ACTOR_H
#define SNN_ACTOR_H

#include "LIF.h"
void actorNetwork(
  fixed_t state[INPUTSIZE],
  fixed_t &action,
  fixed_t w1[HIDDENDIM1][INPUTSIZE],
  fixed_t w2[HIDDENDIM2][HIDDENDIM1],
  fixed_t w3[OUTPUTSIZE][HIDDENDIM2],
  fixed_t leak=2.,
  fixed_t threshold=1.
);

#endif // !SNN_ACTOR_H
