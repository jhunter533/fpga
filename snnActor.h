#ifndef SNN_ACTOR_H
#define SNN_ACTOR_H

#include "LIF.h"

void layer1Forward(fixed_t state[INPUTSIZE], fixed_t w1[HIDDENDIM1][INPUTSIZE],
                   fixed_t mem1[HIDDENDIM1], fixed_t spikes1[HIDDENDIM1],
                   fixed_t leak, fixed_t threshold);

void layer2Forward(fixed_t input[HIDDENDIM1],
                   fixed_t w2[HIDDENDIM2][HIDDENDIM1], fixed_t mem2[HIDDENDIM2],
                   fixed_t spikes2[HIDDENDIM2], fixed_t leak,
                   fixed_t threshold);

void outputForward(fixed_t input[HIDDENDIM2],
                   fixed_t w3[OUTPUTSIZE][HIDDENDIM2],
                   fixed_t memOut[OUTPUTSIZE], fixed_t leak);

void outputBackward(fixed_t memOutD, fixed_t w3[OUTPUTSIZE][HIDDENDIM2],
                    fixed_t w3D[OUTPUTSIZE][HIDDENDIM2],
                    fixed_t spikeIn3D[HIDDENDIM2], fixed_t spikes2[HIDDENDIM2]);

void layer1Backward(fixed_t spikes2D[HIDDENDIM2],
                    fixed_t w2[HIDDENDIM2][HIDDENDIM1],
                    fixed_t mem1[HIDDENDIM1], fixed_t threshold,
                    fixed_t state[INPUTSIZE], fixed_t spikes1D[HIDDENDIM1],
                    fixed_t w1D[HIDDENDIM1][INPUTSIZE]);

void layer2Backward(fixed_t spikeIn3D[HIDDENDIM2], fixed_t mem2[HIDDENDIM2],
                    fixed_t threshold, fixed_t spikes1[HIDDENDIM1],
                    fixed_t w2[HIDDENDIM2][HIDDENDIM1],
                    fixed_t spikes2D[HIDDENDIM2],
                    fixed_t w2D[HIDDENDIM2][HIDDENDIM1]);

void updateWeights(fixed_t w1[HIDDENDIM1][INPUTSIZE],
                   fixed_t w1D[HIDDENDIM1][INPUTSIZE],
                   fixed_t w2[HIDDENDIM2][HIDDENDIM1],
                   fixed_t w2D[HIDDENDIM2][HIDDENDIM1],
                   fixed_t w3[OUTPUTSIZE][HIDDENDIM2],
                   fixed_t w3D[OUTPUTSIZE][HIDDENDIM2], fixed_t learningRate);

void actorForward(fixed_t state[INPUTSIZE], fixed_t &action,
                  fixed_t w1[HIDDENDIM1][INPUTSIZE],
                  fixed_t w2[HIDDENDIM2][HIDDENDIM1],
                  fixed_t w3[OUTPUTSIZE][HIDDENDIM2], fixed_t mem1[HIDDENDIM1],
                  fixed_t mem2[HIDDENDIM2], fixed_t memOut[OUTPUTSIZE],
                  fixed_t spikes1[HIDDENDIM1], fixed_t spikes2[HIDDENDIM2],
                  fixed_t leak, fixed_t threshold);

void actorBackward(fixed_t state[INPUTSIZE], fixed_t dAction,
                   fixed_t w1[HIDDENDIM1][INPUTSIZE],
                   fixed_t w2[HIDDENDIM2][HIDDENDIM1],
                   fixed_t w3[OUTPUTSIZE][HIDDENDIM2], fixed_t mem1[HIDDENDIM1],
                   fixed_t mem2[HIDDENDIM2], fixed_t spikes1[HIDDENDIM1],
                   fixed_t spikes2[HIDDENDIM2],
                   fixed_t w1D[HIDDENDIM1][INPUTSIZE],
                   fixed_t w2D[HIDDENDIM2][HIDDENDIM1],
                   fixed_t w3D[OUTPUTSIZE][HIDDENDIM2], fixed_t leak,
                   fixed_t threshold, fixed_t learningRate);

#endif // !SNN_ACTOR_H
