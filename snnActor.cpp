#include "snnActor.h"
#include "LIF.h"

void layer1Forward(fixed_t state[INPUTSIZE], fixed_t w1[HIDDENDIM1][INPUTSIZE],
                   fixed_t mem1[HIDDENDIM1], fixed_t spikes1[HIDDENDIM1],
                   fixed_t leak, fixed_t threshold) {
#pragma HLS INLINE
LAYER1:
  for (int n = 0; n < HIDDENDIM1; n++) {
#pragma HLS PIPELINE II = 1
    lifNeuron(state, w1[n], mem1[n], spikes1[n], leak, threshold);
  }
}
void layer2Forward(fixed_t input[HIDDENDIM1],
                   fixed_t w2[HIDDENDIM2][HIDDENDIM1], fixed_t mem2[HIDDENDIM2],
                   fixed_t spikes2[HIDDENDIM2], fixed_t leak,
                   fixed_t threshold) {
#pragma HLS INLINE
LAYER2:
  for (int n = 0; n < HIDDENDIM2; n++) {
#pragma HLS PIPELINE II = 8
    lifNeuron(input, w2[n], mem2[n], spikes2[n], leak, threshold);
  }
}
void outputForward(fixed_t input[HIDDENDIM2],
                   fixed_t w3[OUTPUTSIZE][HIDDENDIM2],
                   fixed_t memOut[OUTPUTSIZE], fixed_t leak) {
#pragma HLS INLINE
  outputNeuron(input, w3[0], memOut[0], leak);
}
void outputBackward(fixed_t memOutD, fixed_t w3[OUTPUTSIZE][HIDDENDIM2],
                    fixed_t w3D[OUTPUTSIZE][HIDDENDIM2],
                    fixed_t spikeIn3D[HIDDENDIM2],
                    fixed_t spikes2[HIDDENDIM2]) {
#pragma HLS INLINE
OUTPUT_GRAD:
  for (int i = 0; i < HIDDENDIM2; i++) {
#pragma HLS PIPELINE II = 1
    w3D[0][i] = memOutD * spikes2[i];
    spikeIn3D[i] = memOutD * w3[0][i];
  }
}
void layer1Backward(fixed_t spikes2D[HIDDENDIM2],
                    fixed_t w2[HIDDENDIM2][HIDDENDIM1],
                    fixed_t mem1[HIDDENDIM1], fixed_t threshold,
                    fixed_t state[INPUTSIZE], fixed_t spikes1D[HIDDENDIM1],
                    fixed_t w1D[HIDDENDIM1][INPUTSIZE]) {
#pragma HLS INLINE
LAYER1_GRAD:
  for (int n = 0; n < HIDDENDIM1; n++) {
#pragma HLS PIPELINE II = 1
    fixed_t surrogateD = sigmoidSurrogate(mem1[n] - threshold);
    fixed_t gradAccum = 0;
    for (int j = 0; j < HIDDENDIM2; j++) {
#pragma HLS UNROLL factor = 4
      gradAccum += spikes2D[j] * w2[j][n];
    }
    spikes1D[n] = gradAccum * surrogateD;
    for (int i = 0; i < INPUTSIZE; i++) {
#pragma HLS UNROLL
      w1D[n][i] = spikes1D[n] * state[i];
    }
  }
}
void layer2Backward(fixed_t spikeIn3D[HIDDENDIM2], fixed_t mem2[HIDDENDIM2],
                    fixed_t threshold, fixed_t spikes1[HIDDENDIM1],
                    fixed_t w2[HIDDENDIM2][HIDDENDIM1],
                    fixed_t spikes2D[HIDDENDIM2],
                    fixed_t w2D[HIDDENDIM2][HIDDENDIM1]) {
#pragma HLS INLINE
LAYER2_GRAD:
  for (int n = 0; n < HIDDENDIM2; n++) {
#pragma HLS PIPELINE II = 8
    fixed_t surrogateD = sigmoidSurrogate(mem2[n] - threshold);
    spikes2D[n] = spikeIn3D[n] * surrogateD;
    for (int i = 0; i < HIDDENDIM1; i++) {
#pragma HLS UNROLL factor = 4
      w2D[n][i] = spikes2D[n] * spikes1[i];
    }
  }
}
void updateWeights(fixed_t w1[HIDDENDIM1][INPUTSIZE],
                   fixed_t w1D[HIDDENDIM1][INPUTSIZE],
                   fixed_t w2[HIDDENDIM2][HIDDENDIM1],
                   fixed_t w2D[HIDDENDIM2][HIDDENDIM1],
                   fixed_t w3[OUTPUTSIZE][HIDDENDIM2],
                   fixed_t w3D[OUTPUTSIZE][HIDDENDIM2], fixed_t learningRate) {
#pragma HLS INLINE
UPDATEW1:
  for (int n = 0; n < HIDDENDIM1; n++) {
#pragma HLS PIPELINE II = 1
    for (int i = 0; i < INPUTSIZE; i++) {
      w1[n][i] -= learningRate * w1D[n][i];
    }
  }
UPDATEW2:
  for (int n = 0; n < HIDDENDIM2; n++) {
#pragma HLS PIPELINE II = 8
    for (int i = 0; i < HIDDENDIM1; i++) {
#pragma HLS UNROLL factor = 4
      w2[n][i] -= learningRate * w2D[n][i];
    }
  }
UPDATEW3:
  for (int i = 0; i < HIDDENDIM2; i++) {
#pragma HLS PIPELINE II = 1
    w3[0][i] -= learningRate * w3D[0][i];
  }
}
void actorForward(fixed_t state[INPUTSIZE], fixed_t &action,
                  fixed_t w1[HIDDENDIM1][INPUTSIZE],
                  fixed_t w2[HIDDENDIM2][HIDDENDIM1],
                  fixed_t w3[OUTPUTSIZE][HIDDENDIM2], fixed_t mem1[HIDDENDIM1],
                  fixed_t mem2[HIDDENDIM2], fixed_t memOut[OUTPUTSIZE],
                  fixed_t spikes1[HIDDENDIM1], fixed_t spikes2[HIDDENDIM2],
                  fixed_t leak, fixed_t threshold) {
#pragma HLS DATAFLOW
#pragma HLS ARRAY_PARTITION variable = w1 cyclic factor = 16 dim = 1
#pragma HLS ARRAY_PARTITION variable = w2 cyclic factor = 8 dim = 1
#pragma HLS ARRAY_PARTITION variable = w3 complete dim = 0
#pragma HLS ARRAY_PARTITION variable = mem1 cyclic factor = 16
#pragma HLS ARRAY_PARTITION variable = mem2 cyclic factor = 8
#pragma HLS ARRAY_PARTITION variable = memOut complete
#pragma HLS ARRAY_PARTITION variable = spikes1 cyclic factor = 16
#pragma HLS ARRAY_PARTITION variable = spikes2 cyclic factor = 8
#pragma HLS ARRAY_PARTITION variable = state complete

  layer1Forward(state, w1, mem1, spikes1, leak, threshold);
  layer2Forward(spikes1, w2, mem2, spikes2, leak, threshold);
  outputForward(spikes2, w3, memOut, leak);
  action = 4.0 * memOut[0] - 2.0;
}

void actorBackward(fixed_t state[INPUTSIZE], fixed_t dAction,
                   fixed_t w1[HIDDENDIM1][INPUTSIZE],
                   fixed_t w2[HIDDENDIM2][HIDDENDIM1],
                   fixed_t w3[OUTPUTSIZE][HIDDENDIM2], fixed_t mem1[HIDDENDIM1],
                   fixed_t mem2[HIDDENDIM2], fixed_t spikes1[HIDDENDIM1],
                   fixed_t spikes2[HIDDENDIM2],
                   fixed_t w1D[HIDDENDIM1][INPUTSIZE],
                   fixed_t w2D[HIDDENDIM2][HIDDENDIM1],
                   fixed_t w3D[OUTPUTSIZE][HIDDENDIM2], fixed_t leak,
                   fixed_t threshold, fixed_t learningRate) {
#pragma HLS ARRAY_PARTITION variable = w1 cyclic factor = 16 dim = 1
#pragma HLS ARRAY_PARTITION variable = w2 cyclic factor = 8 dim = 1
#pragma HLS ARRAY_PARTITION variable = w3 complete dim = 0
#pragma HLS ARRAY_PARTITION variable = mem1 cyclic factor = 16
#pragma HLS ARRAY_PARTITION variable = mem2 cyclic factor = 8
#pragma HLS ARRAY_PARTITION variable = memOut complete
#pragma HLS ARRAY_PARTITION variable = spikes1 cyclic factor = 16
#pragma HLS ARRAY_PARTITION variable = spikes2 cyclic factor = 8
#pragma HLS ARRAY_PARTITION variable = state complete
#pragma HLS ARRAY_PARTITION variable = w1D cyclic factor = 16 dim = 1
#pragma HLS ARRAY_PARTITION variable = w2D cyclic factor = 8 dim = 1
#pragma HLS ARRAY_PARTITION variable = w3D complete dim = 0

  fixed_t memOutD = dAction * 4.0;
  fixed_t spikeIn3D[HIDDENDIM2] = {0};
  fixed_t spikes2D[HIDDENDIM2] = {0};
  fixed_t spikes1D[HIDDENDIM1] = {0};

  outputBackward(memOutD, w3, w3D, spikeIn3D, spikes2);
  layer2Backward(spikeIn3D, mem2, threshold, spikes1, w2, spikes2D, w2D);
  layer1Backward(spikes2D, w2, mem1, threshold, state, spikes1D, w1D);
  updateWeights(w1, w1D, w2, w2D, w3, w3D, learningRate);
}
