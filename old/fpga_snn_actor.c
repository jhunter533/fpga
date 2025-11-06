#include "lwip/init.h"
#include "lwip/netif.h"
#include "lwip/tcp.h"
#include "lwip/timeouts.h"
#include "netif/xadapter.h"
#include "platform.h"
#include "platform_config.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "xparameters.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

extern volatile int TcpFastTmrFlag;
extern volatile int TcpSlowTmrFlag;
static struct netif server_netif;
struct netif *app_netif;

void tcp_fasttmr(void);
void tcp_slowtmr(void);

#define TCP_PORT 12345
#define NUM_STATES 24
#define HIDDEN1_SIZE 128
#define HIDDEN2_SIZE 128
#define NUM_ACTIONS 4
#define TIME_STEPS 16

// Fixed-point arithmetic
#define FIXED_POINT_SCALE 1000.0f
#define FIXED_TO_FLOAT(x) ((float)(x) / FIXED_POINT_SCALE)
#define FLOAT_TO_FIXED(x) ((int)((x) * FIXED_POINT_SCALE))

// LIF Neuron parameters
#define V_THRESHOLD 1000 // Fixed point: 1.0
#define V_RESET 0
#define TAU_MEMBRANE 20000 // Fixed point: 20.0
#define DT 1000            // Fixed point: 1.0

// Training parameters
#define LEARNING_RATE 300 // Fixed point: 0.3
#define TAU_TARGET 5      // Fixed point: 0.005

// Network weights
static int fc1_weights[HIDDEN1_SIZE][NUM_STATES];
static int fc2_weights[HIDDEN2_SIZE][HIDDEN1_SIZE];
static int mu_weights[NUM_ACTIONS][HIDDEN2_SIZE];
static int logstd_weights[NUM_ACTIONS][HIDDEN2_SIZE];

static int fc1_bias[HIDDEN1_SIZE];
static int fc2_bias[HIDDEN2_SIZE];
static int mu_bias[NUM_ACTIONS];
static int logstd_bias[NUM_ACTIONS];

// Gradients (for local updates based on critic feedback)
static int fc1_grad[HIDDEN1_SIZE][NUM_STATES];
static int fc2_grad[HIDDEN2_SIZE][HIDDEN1_SIZE];
static int mu_grad[NUM_ACTIONS][HIDDEN2_SIZE];
static int logstd_grad[NUM_ACTIONS][HIDDEN2_SIZE];

// Sigmoid surrogate gradient function (in fixed point)
float surrogate_gradient_sigmoid(float x) {
  // Sigmoid: 1 / (1 + exp(-x))
  // Gradient: sigmoid(x) * (1 - sigmoid(x))
  float sigmoid_x = 1.0f / (1.0f + expf(-x));
  return sigmoid_x * (1.0f - sigmoid_x);
}

// Fast tanh approximation
float fast_tanh(float x) {
  if (x > 2.0f)
    return 1.0f;
  if (x < -2.0f)
    return -1.0f;
  return x - (x * x * x) / 3.0f + (x * x * x * x * x) / 5.0f;
}

// Parallel matrix multiplication using crossbar-like approach
void parallel_matrix_multiply(const int *weights, const float *input,
                              int *output, int rows, int cols) {
  // Process in parallel chunks
  for (int i = 0; i < rows; i++) {
    int sum = 0;
    for (int j = 0; j < cols; j++) {
      sum += (int)(weights[i * cols + j] * FLOAT_TO_FIXED(input[j]));
    }
    output[i] = sum;
  }
}

// LIF neuron structure
typedef struct {
  int v_membrane;
  int v_threshold;
  int tau_membrane;
} lif_neuron_t;

void init_lif_neuron(lif_neuron_t *neuron) {
  neuron->v_membrane = V_RESET;
  neuron->v_threshold = V_THRESHOLD;
  neuron->tau_membrane = TAU_MEMBRANE;
}

int update_lif_neuron_with_gradient(lif_neuron_t *neuron, int current_input,
                                    float *gradient) {
  int dv_dt =
      (0 - neuron->v_membrane) / (neuron->tau_membrane / DT) + current_input;
  neuron->v_membrane += dv_dt;

  if (neuron->v_membrane >= neuron->v_threshold) {
    neuron->v_membrane = V_RESET;
    *gradient = surrogate_gradient_sigmoid(FIXED_TO_FLOAT(neuron->v_membrane));
    return 1; // Spike
  } else {
    *gradient = surrogate_gradient_sigmoid(FIXED_TO_FLOAT(neuron->v_membrane));
    return 0; // No spike
  }
}

// SNN forward pass with surrogate gradients
void snn_forward_pass(const float *state, float *action, float *log_prob) {
  static lif_neuron_t lif1_neurons[HIDDEN1_SIZE];
  static lif_neuron_t lif2_neurons[HIDDEN2_SIZE];
  static float lif1_gradients[HIDDEN1_SIZE];
  static float lif2_gradients[HIDDEN2_SIZE];

  int hidden1[HIDDEN1_SIZE];
  int hidden2[HIDDEN2_SIZE];
  int mu_output[NUM_ACTIONS];
  int logstd_output[NUM_ACTIONS];

  // Initialize neurons
  for (int i = 0; i < HIDDEN1_SIZE; i++) {
    init_lif_neuron(&lif1_neurons[i]);
    lif1_gradients[i] = 0.0f;
  }
  for (int i = 0; i < HIDDEN2_SIZE; i++) {
    init_lif_neuron(&lif2_neurons[i]);
    lif2_gradients[i] = 0.0f;
  }

  // Simulate SNN over time steps
  for (int t = 0; t < TIME_STEPS; t++) {
    // Layer 1: state -> hidden1 (parallel computation)
    parallel_matrix_multiply((int *)fc1_weights, state, hidden1, HIDDEN1_SIZE,
                             NUM_STATES);
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
      update_lif_neuron_with_gradient(
          &lif1_neurons[i], hidden1[i] + fc1_bias[i], &lif1_gradients[i]);
    }

    // Layer 2: hidden1 -> hidden2 (parallel computation)
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
      int sum = fc2_bias[i];
      for (int j = 0; j < HIDDEN1_SIZE; j++) {
        sum += lif1_neurons[j].v_membrane * fc2_weights[i][j];
      }
      update_lif_neuron_with_gradient(&lif2_neurons[i], sum,
                                      &lif2_gradients[i]);
    }
  }

  // Output layers (non-spiking, parallel computation)
  for (int i = 0; i < NUM_ACTIONS; i++) {
    int sum_mu = mu_bias[i];
    int sum_logstd = logstd_bias[i];

    for (int j = 0; j < HIDDEN2_SIZE; j++) {
      sum_mu += lif2_neurons[j].v_membrane * mu_weights[i][j];
      sum_logstd += lif2_neurons[j].v_membrane * logstd_weights[i][j];
    }

    mu_output[i] = sum_mu;
    logstd_output[i] = sum_logstd;
  }

  // Generate action with tanh activation
  for (int i = 0; i < NUM_ACTIONS; i++) {
    float mu_val = FIXED_TO_FLOAT(mu_output[i]);
    float logstd_val = FIXED_TO_FLOAT(logstd_output[i]);

    // Clamp log_std
    if (logstd_val < -20.0f)
      logstd_val = -20.0f;
    if (logstd_val > 2.0f)
      logstd_val = 2.0f;

    // Generate action (simplified - no noise in hardware)
    action[i] = fast_tanh(mu_val);
    log_prob[i] = logstd_val;
  }
}

// Local gradient update based on critic feedback (simplified)
void update_weights_with_gradient(const float *critic_gradient,
                                  float learning_rate) {
  // This is a simplified example - in real implementation,
  // critic gradients would be sent from PC to FPGA

  // Example: update output layer weights based on gradient feedback
  for (int i = 0; i < NUM_ACTIONS; i++) {
    for (int j = 0; j < HIDDEN2_SIZE; j++) {
      // Update mu weights
      int grad_update =
          FLOAT_TO_FIXED(critic_gradient[i] * 0.001f); // Simplified
      mu_weights[i][j] += grad_update;

      // Update logstd weights
      logstd_weights[i][j] += grad_update;
    }
  }
}

void initialize_weights() {
  for (int i = 0; i < HIDDEN1_SIZE; i++) {
    for (int j = 0; j < NUM_STATES; j++) {
      fc1_weights[i][j] =
          FLOAT_TO_FIXED((float)simple_rand() / 32767.0f * 0.2f - 0.1f);
    }
    fc1_bias[i] = 0;
  }

  for (int i = 0; i < HIDDEN2_SIZE; i++) {
    for (int j = 0; j < HIDDEN1_SIZE; j++) {
      fc2_weights[i][j] =
          FLOAT_TO_FIXED((float)simple_rand() / 32767.0f * 0.2f - 0.1f);
    }
    fc2_bias[i] = 0;
  }

  for (int i = 0; i < NUM_ACTIONS; i++) {
    for (int j = 0; j < HIDDEN2_SIZE; j++) {
      mu_weights[i][j] =
          FLOAT_TO_FIXED((float)simple_rand() / 32767.0f * 0.2f - 0.1f);
      logstd_weights[i][j] =
          FLOAT_TO_FIXED((float)simple_rand() / 32767.0f * 0.2f - 0.1f);
    }
    mu_bias[i] = 0;
    logstd_bias[i] = 0;
  }
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

  // Parse incoming data (state from PC)
  if (p->tot_len >= sizeof(float) * (NUM_STATES + 1)) { // state + done flag
    float *data = (float *)p->payload;
    float state[NUM_STATES];
    int done = (int)data[NUM_STATES];

    for (int i = 0; i < NUM_STATES; i++) {
      state[i] = data[i];
    }

    if (done) {
      xil_printf("Episode done received\r\n");
    } else {
      // Forward pass through SNN
      float action[NUM_ACTIONS];
      float log_prob = 0.0f; // Single log_prob value

      snn_forward_pass(state, action, &log_prob);

      // Send action and log_prob back to PC
      char response[sizeof(float) * (NUM_ACTIONS + 1)];
      memcpy(response, action, sizeof(float) * NUM_ACTIONS);
      memcpy(response + sizeof(float) * NUM_ACTIONS, &log_prob, sizeof(float));

      err_t write_err =
          tcp_write(tpcb, response, sizeof(response), TCP_WRITE_FLAG_COPY);
      if (write_err == ERR_OK) {
        tcp_output(tpcb);
      }
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
  xil_printf("----- SNN Actor Server (Port %d) -----\r\n", TCP_PORT);
  xil_printf("FPGA SNN Actor with surrogate gradients\r\n");
  xil_printf("--------------------------------------\r\n");
}

int main() {
  /* IP configuration */
  ip_addr_t ipaddr, netmask, gw;
  IP4_ADDR(&ipaddr, 192, 168, 1, 10);
  IP4_ADDR(&netmask, 255, 255, 255, 0);
  IP4_ADDR(&gw, 192, 168, 1, 1);

  app_netif = &server_netif;

  init_platform();

  print_app_header();

  /* Initialize lwIP */
  lwip_init();

  /* Add network interface */
  if (!xemac_add(app_netif, &ipaddr, &netmask, &gw,
                 (unsigned char[]){0x00, 0x0a, 0x35, 0x00, 0x01, 0x02},
                 PLATFORM_EMAC_BASEADDR)) {
    xil_printf("Error adding network interface\r\n");
    return -1;
  }

  netif_set_default(app_netif);
  netif_set_up(app_netif);

  xil_printf("Network interface is UP.\r\n");

  /* Initialize weights */
  initialize_weights();
  xil_printf("SNN weights initialized.\r\n");

  /* Start TCP server */
  if (init_tcp_server() != 0) {
    xil_printf("Failed to initialize TCP server\r\n");
    return -1;
  }

  xil_printf("SNN Actor ready. Waiting for connections...\r\n");

  /* Main loop */
  while (1) {
    /* Handle lwIP timers */
    if (TcpFastTmrFlag) {
      tcp_fasttmr();
      TcpFastTmrFlag = 0;
    }
    if (TcpSlowTmrFlag) {
      tcp_slowtmr();
      TcpSlowTmrFlag = 0;
    }

    /* Poll for incoming packets */
    xemacif_input(app_netif);

    /* Small delay */
    usleep(1000); // 1ms
  }

  cleanup_platform();
  return 0;
}
