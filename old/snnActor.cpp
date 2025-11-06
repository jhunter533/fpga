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

void muForward(fixed_t input[HIDDENDIM2], fixed_t w3Mu[OUTPUTSIZE][HIDDENDIM2],
               fixed_t &memOutMu, fixed_t leak) {
#pragma HLS PIPELINE II = 1
  outputNeuron(input, w3Mu[0], memOutMu, leak);
}
void logForward(fixed_t input[HIDDENDIM2],
                fixed_t w3Log[OUTPUTSIZE][HIDDENDIM2], fixed_t &memOutLog,
                fixed_t leak) {
#pragma HLS PIPELINE II = 1
  outputNeuron(input, w3Log[0], memOutLog, leak);
}
void muBackward(fixed_t memOutDMu, fixed_t w3Mu[OUTPUTSIZE][HIDDENDIM2],
                fixed_t w3DMu[OUTPUTSIZE][HIDDENDIM2],
                fixed_t spikeIn3D[HIDDENDIM2], fixed_t spikes2[HIDDENDIM2]) {
#pragma HLS PIPELINE II = 1
  fixed_t surrogateG = sigmoidSurrogate(memOutDMu);
  for (int i = 0; i < HIDDENDIM2; i++) {
    w3DMu[0][i] = memOutDMu * surrogateG * spikes2[i];
    spikeIn3D[i] += memOutDMu * surrogateG * w3Mu[0][i];
  }
}
void logBackward(fixed_t memOutDLog, fixed_t w3Log[OUTPUTSIZE][HIDDENDIM2],
                 fixed_t w3DLog[OUTPUTSIZE][HIDDENDIM2],
                 fixed_t spikeIn3D[HIDDENDIM2], fixed_t spikes2[HIDDENDIM2]) {
#pragma HLS PIPELINE II = 1
  fixed_t surrogateG = sigmoidSurrogate(memOutDLog);
  for (int i = 0; i < HIDDENDIM2; i++) {
    w3DLog[0][i] = memOutDLog * surrogateG * spikes2[i];
    spikeIn3D += memOutDLog * surrogateG * w3Log[0][i];
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
void updateWeights(
    fixed_t w1[HIDDENDIM1][INPUTSIZE], fixed_t w1D[HIDDENDIM1][INPUTSIZE],
    fixed_t w2[HIDDENDIM2][HIDDENDIM1], fixed_t w2D[HIDDENDIM2][HIDDENDIM1],
    fixed_t w3Mu[OUTPUTSIZE][HIDDENDIM2], fixed_t w3DMu[OUTPUTSIZE][HIDDENDIM2],
    fixed_t w3Log[OUTPUTSIZE][HIDDENDIM2],
    fixed_t w3DLog[OUTPUTSIZE][HIDDENDIM2], fixed_t learningRate) {
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
UPDATEW3MU:
  for (int i = 0; i < HIDDENDIM2; i++) {
#pragma HLS PIPELINE II = 1
    w3Mu[0][i] -= learningRate * w3DMu[0][i];
  }
UPDATEW3LOG:
  for (int i = 0; i < HIDDENDIM2; i++) {
#pragma HLS PIPELINE II = 1
    w3Log[0][i] -= learningRate * w3DLog[0][i];
  }
}
void actorForward(fixed_t state[INPUTSIZE], fixed_t &mu, fixed_t &log_std,
                  fixed_t w1[HIDDENDIM1][INPUTSIZE],
                  fixed_t w2[HIDDENDIM2][HIDDENDIM1],
                  fixed_t w3Mu[OUTPUTSIZE][HIDDENDIM2],
                  fixed_t w3Log[OUTPUTSIZE][HIDDENDIM2],
                  fixed_t mem1[HIDDENDIM1], fixed_t mem2[HIDDENDIM2],
                  fixed_t &memOutMu, fixed_t &memOutLog,
                  fixed_t spikes1[HIDDENDIM1], fixed_t spikes2[HIDDENDIM2],
                  fixed_t leak, fixed_t threshold) {
#pragma HLS DATAFLOW
#pragma HLS ARRAY_PARTITION variable = w1 cyclic factor = 16 dim = 2
#pragma HLS ARRAY_PARTITION variable = w2 cyclic factor = 8 dim = 2
#pragma HLS ARRAY_PARTITION variable = w3Mu complete dim = 0
#pragma HLS ARRAY_PARTITION variable = w3Log complete dim = 0
#pragma HLS ARRAY_PARTITION variable = mem1 cyclic factor = 16
#pragma HLS ARRAY_PARTITION variable = mem2 cyclic factor = 8
#pragma HLS ARRAY_PARTITION variable = memOutMu complete
#pragma HLS ARRAY_PARTITION variable = memOutLog complete
#pragma HLS ARRAY_PARTITION variable = spikes1 cyclic factor = 16
#pragma HLS ARRAY_PARTITION variable = spikes2 cyclic factor = 8
#pragma HLS ARRAY_PARTITION variable = state complete

  layer1Forward(state, w1, mem1, spikes1, leak, threshold);
  layer2Forward(spikes1, w2, mem2, spikes2, leak, threshold);
  muForward(spikes2, w3Mu, memOutMu, leak);
  logForward(spikes2, w3Log, memOutLog, leak);
  mu = fixed_t(4.0) * memOutMu - fixed_t(2.0);
  log_std = clampLog(memOutLog);
}

void actorBackward(fixed_t state[INPUTSIZE], fixed_t dMu, fixed_t dLog,
                   fixed_t w1[HIDDENDIM1][INPUTSIZE],
                   fixed_t w2[HIDDENDIM2][HIDDENDIM1],
                   fixed_t w3Mu[OUTPUTSIZE][HIDDENDIM2],
                   fixed_t w3Log[OUTPUTSIZE][HIDDENDIM2],
                   fixed_t mem1[HIDDENDIM1], fixed_t mem2[HIDDENDIM2],
                   fixed_t spikes1[HIDDENDIM1], fixed_t spikes2[HIDDENDIM2],
                   fixed_t w1D[HIDDENDIM1][INPUTSIZE],
                   fixed_t w2D[HIDDENDIM2][HIDDENDIM1],
                   fixed_t w3DMu[OUTPUTSIZE][HIDDENDIM2],
                   fixed_t w3DLog[OUTPUTSIZE][HIDDENDIM2], fixed_t leak,
                   fixed_t threshold, fixed_t learningRate) {
#pragma HLS ARRAY_PARTITION variable = w1 cyclic factor = 16 dim = 1
#pragma HLS ARRAY_PARTITION variable = w2 cyclic factor = 8 dim = 1
#pragma HLS ARRAY_PARTITION variable = w3Mu complete dim = 0
#pragma HLS ARRAY_PARTITION variable = w3Log complete dim = 0
#pragma HLS ARRAY_PARTITION variable = mem1 cyclic factor = 16
#pragma HLS ARRAY_PARTITION variable = mem2 cyclic factor = 8
#pragma HLS ARRAY_PARTITION variable = spikes1 cyclic factor = 16
#pragma HLS ARRAY_PARTITION variable = spikes2 cyclic factor = 8
#pragma HLS ARRAY_PARTITION variable = state complete
#pragma HLS ARRAY_PARTITION variable = w1D cyclic factor = 16 dim = 1
#pragma HLS ARRAY_PARTITION variable = w2D cyclic factor = 8 dim = 1
#pragma HLS ARRAY_PARTITION variable = w3DMu complete dim = 0
#pragma HLS ARRAY_PARTITION variable = w3DLog complete dim = 0
  fixed_t memOutDMu = fixed_t(dMu) * fixed_t(4.0);
  fixed_t memOutDLog = dLog;
  fixed_t spikeIn3D[HIDDENDIM2] = {0};
  muBackward(memOutDMu, w3Mu, w3DMu, spikeIn3D, spikes2);
  logBackward(memOutDLog, w3Log, w3DLog, spikeIn3D, spikes2);
  fixed_t spikes2D[HIDDENDIM2] = {0};
  fixed_t w2DTemp[HIDDENDIM2][HIDDENDIM1] = {0};
  layer2Backward(spikeIn3D, mem2, threshold, spikes1, w2, spikes2D, w2DTemp);

  fixed_t spikes1D[HIDDENDIM1] = {0};
  fixed_t w1DTemp[HIDDENDIM1][INPUTSIZE] = {0};
  layer1Backward(spikes2D, w2, mem1, threshold, state, spikes1D, w1DTemp);

  for (int n = 0; n < HIDDENDIM1; n++) {
    for (int i = 0; i < INPUTSIZE; i++) {
      w1D[n][i] = w1DTemp[n][i];
    }
  }
  for (int n = 0; n < HIDDENDIM2; n++) {
    for (int i = 0; i < HIDDENDIM1; i++) {
      w2D[n][i] = w2DTemp[n][i];
    }
  }
}
