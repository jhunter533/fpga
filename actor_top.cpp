#include "LIF.h"
#include "snnActor.h"
#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <memory>
#include <string.h>

#define DWIDTH 32
typedef ap_axiu<DWIDTH, 0, 0, 0> axis_t;

void fpgaActor(hls::stream<axis_t> &inStream, hls::stream<axis_t> &outStream) {

#pragma HLS INTERFACE axis port = inStream
#pragma HLS INTERFACE axis port = outStream
#pragma HLS INTERFACE s_axilite port = return

  static fixed_t leak = 0.5;
  static fixed_t threshold = 1.0;

  static fixed_t mem1[HIDDENDIM1] = {0};
  static fixed_t mem2[HIDDENDIM2] = {0};
  static fixed_t memOut[OUTPUTSIZE] = {0};
  static fixed_t spikes1[HIDDENDIM1] = {0};
  static fixed_t spikes2[HIDDENDIM2] = {0};

  static fixed_t w1[HIDDENDIM1][INPUTSIZE];
  static fixed_t w2[HIDDENDIM2][HIDDENDIM1];
  static fixed_t w3[OUTPUTSIZE][HIDDENDIM2];

  static bool weightsInit = false;
  if (!weightsInit) {
    weightsInit = true;
    for (int i = 0; i < HIDDENDIM1; i++) {
      for (int j = 0; j < INPUTSIZE; j++) {
        w1[i][j] = (fixed_t)((rand() % 2000 - 1000) / 1000.0);
      }
    }
  }
#pragma HLS BIND_STORAGE variable = w1 type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = w2 type = ram_2p impl = bram
  axis_t cmdPkt = inStream.read();
  char command = cmdPkt.data;
  if (command == 'F') {
    fixed_t state[INPUTSIZE];
    for (int i = 0; i < INPUTSIZE; i++) {
#pragma HLS PIPELINE II = 1
      axis_t val = inStream.read();
      state[i] = *reinterpret_cast<fixed_t *>(&val.data);
    }
    fixed_t action;
    actorForward(state, action, w1, w2, w3, mem1, mem2, memOut, spikes1,
                 spikes2, leak, threshold);
    axis_t outVal;
    outVal.data = *reinterpret_cast<unsigned *>(&action);
    outStream.write(outVal);
  } else if (command == 'R') {
  RESET_NEURONS:
    for (int i = 0; i < HIDDENDIM1; i++) {
#pragma HLS PIPELINE II = 1
      mem1[i] = 0;
      spikes1[i] = 0;
    }
    for (int i = 0; i < HIDDENDIM2; i++) {
#pragma HLS PIPELINE II = 1
      mem2[i] = 0;
      spikes2[i] = 0;
    }
    for (int i = 0; i < OUTPUTSIZE; i++) {
#pragma HLS PIPELINE II = 1
      memOut[i] = 0;
    }
    axis_t ack;
    ack.data = 'D';
    outStream.write(ack);
  }
}
