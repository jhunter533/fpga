#include "LIF.h"
#include "snnActor.h"
#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <cstdint>
#include <hls_math.h>
#include <hls_stream.h>
#include <memory>
#include <string.h>
#include <stdio.h>

#define DWIDTH 32
typedef ap_axiu<DWIDTH, 0, 0, 0> axis_t;

static enum { IDLE, CMD, LEN, DATA, CHECKSUM } state = IDLE;

void sendPacket(hls::stream<axis_t> &outStream, uint8_t cmd, uint8_t *data,
                uint8_t len) {
  axis_t tx;
  uint8_t checkSum = 0xAA + cmd + len;

  tx.data = 0xAA;
  tx.last = 0;
  outStream.write(tx);
  checkSum += 0xAA;

  tx.data = cmd;
  outStream.write(tx);
  checkSum += cmd;

  tx.data = len;
  outStream.write(tx);
  checkSum += len;

  for (int i = 0; i < len; i++) {
    tx.data = data[i];
    tx.last = (i == len - 1) ? 1 : 0;
    outStream.write(tx);
    checkSum += data[i];
  }

  tx.data = checkSum;
  tx.last = 1;
  outStream.write(tx);
  // checksum
}
void sendAck(hls::stream<axis_t> &outStream, uint8_t ackCommand) {
  axis_t tx;
  uint8_t len=0;
  uint8_t checksum=(0xAA+ackCommand+len)&0xFF;

  tx.data = 0xAA;
  tx.last = 0;
  outStream.write(tx);

  tx.data=ackCommand;
  tx.last=0;
  outStream.write(tx);

  tx.data=len;
  tx.last=0;
  outStream.write(tx);

  tx.data=checksum;
  tx.last=1;
  outStream.write(tx);
}
void resetNeurons(fixed_t mem1[HIDDENDIM1], fixed_t mem2[HIDDENDIM2],
                  fixed_t &memOutMu, fixed_t &memOutLog,
                  fixed_t spikes1[HIDDENDIM1], fixed_t spikes2[HIDDENDIM2]) {
  for (int i = 0; i < HIDDENDIM1; i++) {
    mem1[i] = 0;
    spikes1[i] = 0;
  }
  for (int i = 0; i < HIDDENDIM2; i++) {
    mem2[i] = 0;
    spikes2[i] = 0;
  }
  memOutMu = 0;
  memOutLog = 0;
}

void processCommand(
    uint8_t cmd, uint8_t *data, uint8_t len, fixed_t last_state[INPUTSIZE],
    hls::stream<axis_t> &outStream, fixed_t w1[HIDDENDIM1][INPUTSIZE],
    fixed_t w2[HIDDENDIM2][HIDDENDIM1], fixed_t w3Mu[OUTPUTSIZE][HIDDENDIM2],
    fixed_t w3Log[OUTPUTSIZE][HIDDENDIM2], fixed_t mem1[HIDDENDIM1],
    fixed_t mem2[HIDDENDIM2], fixed_t &memOutMu, fixed_t &memOutLog,
    fixed_t spikes1[HIDDENDIM1], fixed_t spikes2[HIDDENDIM2]) {
  printf("fpgaActor: processing command 0x%02X,len %d\r\n",cmd,len);
  if (cmd == 0x52) {
    resetNeurons(mem1, mem2, memOutMu, memOutLog, spikes1, spikes2);
    sendAck(outStream, 0x44);
  } else if (cmd == 0x46) {
    fixed_t state[INPUTSIZE];
    for (int i = 0; i < INPUTSIZE; i++) {
      uint16_t fixedVal = (data[2 * i + 1] << 8) | data[2 * i];
      state[i] = *reinterpret_cast<fixed_t *>(&fixedVal);
      last_state[i] = state[i];
    }
    fixed_t mu, log_std;
    actorForward(state, mu, log_std, w1, w2, w3Mu, w3Log, mem1, mem2, memOutMu,
                 memOutLog, spikes1, spikes2, .5, 1.0);
    uint8_t response[4];
    uint16_t mu_fixed = *reinterpret_cast<uint16_t *>(&mu);
    uint16_t log_std_fixed = *reinterpret_cast<uint16_t *>(&log_std);
    response[0] = mu_fixed & 0xFF;
    response[1] = (mu_fixed >> 8) & 0xFF;
    response[2] = log_std_fixed & 0xFF;
    response[3] = (log_std_fixed >> 8) & 0xFF;

    sendPacket(outStream, 0x46, response, 4);
  } else if (cmd == 0x42) {
    uint16_t d_mu_fixed = (data[1] << 8) | data[0];
    uint16_t d_log_std_fixed = (data[3] << 8) | data[2];
    uint16_t reward_fixed = (data[5] << 8) | data[4];
    fixed_t d_mu = *reinterpret_cast<fixed_t *>(&d_mu_fixed);
    fixed_t d_log_std = *reinterpret_cast<fixed_t *>(&d_log_std_fixed);
    fixed_t reward = *reinterpret_cast<fixed_t *>(&reward_fixed);
    fixed_t w1D[HIDDENDIM1][INPUTSIZE] = {0};
    fixed_t w2D[HIDDENDIM2][HIDDENDIM1] = {0};
    fixed_t w3DMu[OUTPUTSIZE][HIDDENDIM2] = {0};
    fixed_t w3DLog[OUTPUTSIZE][HIDDENDIM2] = {0};

    actorBackward(last_state, d_mu, d_log_std, w1, w2, w3Mu, w3Log, mem1, mem2,
                  spikes1, spikes2, w1D, w2D, w3DMu, w3DLog, .5, 1.0, .001);

    sendAck(outStream, 0x44);
  }
}
void fpgaActor(hls::stream<axis_t> &inStream, hls::stream<axis_t> &outStream) {
#pragma HLS INTERFACE axis port = inStream
#pragma HLS INTERFACE axis port = outStream
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS PIPELINE II = 1
  static uint8_t cmd, len, data[64];
  static uint16_t data_idx = 0;
  static fixed_t last_state[INPUTSIZE] = {0};

  static fixed_t mem1[HIDDENDIM1] = {0};
  static fixed_t mem2[HIDDENDIM2] = {0};
  static fixed_t memOutMu = 0;
  static fixed_t memOutLog = 0;
  static fixed_t spikes1[HIDDENDIM1] = {0};
  static fixed_t spikes2[HIDDENDIM2] = {0};

  static fixed_t w1[HIDDENDIM1][INPUTSIZE];
  static fixed_t w2[HIDDENDIM2][HIDDENDIM1];
  static fixed_t w3Mu[OUTPUTSIZE][HIDDENDIM2];
  static fixed_t w3Log[OUTPUTSIZE][HIDDENDIM2];

#pragma HLS BIND_STORAGE variable = w1 type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = w2 type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = w3Mu type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = w3Log type = ram_2p impl = bram

  if (!inStream.empty()) {
    axis_t rx = inStream.read();
    switch (state) {
    case IDLE:
      if (rx.data == 0xAA)
        state = CMD;
      break;
    case CMD:
      cmd = rx.data;
      state = LEN;
      break;
    case LEN:
      len = rx.data;
      data_idx = 0;
      state = (len > 0) ? DATA : CHECKSUM;
      break;
    case DATA:
      data[data_idx++] = rx.data;
      if (data_idx >= len)
        state = CHECKSUM;
      break;
    case CHECKSUM:
      uint8_t calcChecksum = 0xAA + cmd + len;
      for (int i = 0; i < len; i++)
        calcChecksum += data[i];
      if (calcChecksum == rx.data) {
        processCommand(cmd, data, len, last_state, outStream,w1,w2,w3Mu,w3Log,mem1,mem2,memOutMu,memOutLog,spikes1,spikes2);
      }
      state = IDLE;
      break;
    }
  }
}

