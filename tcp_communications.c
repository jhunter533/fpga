#include "lwip/init.h"
#include "lwip/netif.h"
#include "lwip/tcp.h"
#include "lwip/timeouts.h"
#include "netif/xadapter.h"
#include "platform.h"
#include "platform_config.h"
#include "xil_cache.h"
#include "xil_io.h"
#include "xil_printf.h"
#include "xparameters.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>

extern volatile int TcpFastTmrFlag;
extern volatile int TcpSlowTmrFlag;

struct netif *app_netif;
static struct netif server_netif;
struct netif *echo_netif;

// SNN IP register addresses (from Vivado block design)
#define SNN_BASE_ADDR XPAR_SNN_HLS_KERNEL_0_S_AXI_CONTROL_ADDR
#define SNN_STATE_ADDR (SNN_BASE_ADDR + 0x10)
#define SNN_ACTION_ADDR (SNN_BASE_ADDR + 0x20)
#define SNN_LOGPROB_ADDR (SNN_BASE_ADDR + 0x28)
#define SNN_CTRL_ADDR (SNN_BASE_ADDR + 0x00)

// Network configuration
#define TCP_PORT 12345
#define NUM_STATES 3
#define NUM_ACTIONS 1

// Fixed-point conversion macros
#define FLOAT_TO_FXP(x) ((int)((x) * 1024.0f))
#define FXPTO_FLOAT(x) ((float)(x) / 1024.0f)

// Function to communicate with SNN HLS kernel
void snn_forward_pass_pl(float *state, float *action, float *log_prob) {
  // Write state to SNN IP
  for (int i = 0; i < NUM_STATES; i++) {
    Xil_Out32(SNN_STATE_ADDR + i * 4, FLOAT_TO_FXP(state[i]));
  }

  // Trigger SNN computation
  Xil_Out32(SNN_CTRL_ADDR, 0x01);

  // Wait for completion (check done bit)
  while ((Xil_In32(SNN_CTRL_ADDR) & 0x02) == 0) {
    // Wait for SNN to complete
    usleep(10); // Small delay
  }

  // Read action output
  int action_raw = Xil_In32(SNN_ACTION_ADDR);
  *action = FXPTO_FLOAT(action_raw);

  // Read log_prob output
  int logprob_raw = Xil_In32(SNN_LOGPROB_ADDR);
  *log_prob = FXPTO_FLOAT(logprob_raw);
}

// Function to load weights to SNN IP (when needed)
void load_weights_to_snn(float *weights, int weight_count) {
  // This would be called when updating weights from PC
  // Implementation depends on your weight loading interface
  // For now, placeholder:
  xil_printf("Loading %d weights to SNN IP\r\n", weight_count);
  // Write weights to appropriate BRAM via AXI
}

err_t tcp_data_received(void *arg, struct tcp_pcb *tpcb, struct pbuf *p,
                        err_t err) {
  if (err != ERR_OK || p == NULL) {
    if (p != NULL) {
      pbuf_free(p);
    }
    tcp_close(tpcb);
    return ERR_OK;
  }

  if (p->tot_len >= sizeof(float) * (NUM_STATES + 1)) { // state + done flag
    float *data = (float *)p->payload;
    float state[NUM_STATES];

    for (int i = 0; i < NUM_STATES; i++) {
      state[i] = data[i];
    }
    int done = (int)data[NUM_STATES];

    // Forward pass through SNN (now uses PL)
    float action;
    float log_prob;

    snn_forward_pass_pl(state, &action, &log_prob);

    // Send action and log_prob back to PC
    char response[sizeof(float) * 2]; // action + log_prob
    float response_data[2] = {action, log_prob};
    memcpy(response, response_data, sizeof(response));

    err_t write_err =
        tcp_write(tpcb, response, sizeof(response), TCP_WRITE_FLAG_COPY);
    if (write_err == ERR_OK) {
      tcp_output(tpcb);
    }
  }

  tcp_recved(tpcb, p->tot_len);
  pbuf_free(p);
  return ERR_OK;
}

err_t tcp_connection_accepted(void *arg, struct tcp_pcb *newpcb, err_t err) {
  if (err != ERR_OK || newpcb == NULL) {
    return ERR_VAL;
  }

  xil_printf("PC connected for SNN actor interaction\r\n");

  tcp_arg(newpcb, NULL);
  tcp_recv(newpcb, tcp_data_received);
  return ERR_OK;
}

int init_tcp_server() {
  struct tcp_pcb *pcb;
  pcb = tcp_new();
  if (pcb == NULL) {
    xil_printf("Error: could not create PCB\r\n");
    return -1;
  }

  err_t err = tcp_bind(pcb, IP_ADDR_ANY, TCP_PORT);
  if (err != ERR_OK) {
    xil_printf("Error: could not bind to port %d\r\n", TCP_PORT);
    tcp_close(pcb);
    return -1;
  }

  pcb = tcp_listen(pcb);
  if (pcb == NULL) {
    xil_printf("Error: could not listen\r\n");
    return -1;
  }

  tcp_accept(pcb, tcp_connection_accepted);
  xil_printf("SNN Actor TCP server listening on port %d\r\n", TCP_PORT);
  return 0;
}

void print_app_header() {
  xil_printf("\r\n\n\r\n");
  xil_printf("----- FPGA SNN Actor Server (Port %d) -----\r\n", TCP_PORT);
  xil_printf("PL SNN with AXI communication\r\n");
  xil_printf("--------------------------------------\r\n");
}

int main() {
  // IP configuration
  ip_addr_t ipaddr, netmask, gw;
  IP4_ADDR(&ipaddr, 192, 168, 1, 10);
  IP4_ADDR(&netmask, 255, 255, 255, 0);
  IP4_ADDR(&gw, 192, 168, 1, 1);

  app_netif = &server_netif;

  init_platform();
  print_app_header();

  // Initialize lwIP
  lwip_init();

  // Add network interface
  if (!xemac_add(app_netif, &ipaddr, &netmask, &gw,
                 (unsigned char[]){0x00, 0x0a, 0x35, 0x00, 0x01, 0x02},
                 PLATFORM_EMAC_BASEADDR)) {
    xil_printf("Error adding network interface\r\n");
    return -1;
  }

  netif_set_default(app_netif);
  netif_set_up(app_netif);
  xil_printf("Network interface is UP.\r\n");

  // Start TCP server
  if (init_tcp_server() != 0) {
    xil_printf("Failed to initialize TCP server\r\n");
    return -1;
  }

  xil_printf("SNN Actor ready. Waiting for connections...\r\n");

  // Main loop
  while (1) {
    if (TcpFastTmrFlag) {
      tcp_fasttmr();
      TcpFastTmrFlag = 0;
    }
    if (TcpSlowTmrFlag) {
      tcp_slowtmr();
      TcpSlowTmrFlag = 0;
    }

    xemacif_input(app_netif);
    usleep(1000);
  }

  cleanup_platform();
  return 0;
}
