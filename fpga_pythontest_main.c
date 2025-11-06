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

#include <stdio.h>
#include <string.h>
#include <unistd.h>

extern volatile int TcpFastTmrFlag;
extern volatile int TcpSlowTmrFlag;

struct netif *app_netif;
static struct netif server_netif;
struct netif *echo_netif;
static unsigned int simple_rand_state = 12345;

// to be changeed later dont worry about it right now
int simple_rand() {
  simple_rand_state = simple_rand_state * 1103515245 + 12345;
  return (unsigned int)(simple_rand_state / 65536) % 32768;
}

void tcp_fasttmr(void);
void tcp_slowtmr(void);

#define TCP_PORT 12345
#define RAND_MAX 32767.0f
#define NUM_STATES 3
#define HIDDEN1_SIZE 128
#define HIDDEN2_SIZE 128
#define NUM_ACTIONS 1
#define TIME_STEPS 2

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

static float hidden1_trace[TIME_STEPS][HIDDEN1_SIZE];
static float hidden2_trace[TIME_STEPS][HIDDEN2_SIZE];
static float lif1_spikes[TIME_STEPS][HIDDEN1_SIZE];
static float lif2_spikes[TIME_STEPS][HIDDEN2_SIZE];

float fast_exp(float x) {
  if (x > 10.0f)
    return 22026.46579f; // e^10
  if (x < -10.0f)
    return 0.000045399f; // e^-10

  float result = 1.0f;
  float term = 1.0f;
  for (int i = 1; i <= 10; i++) {
    term = term * x / i;
    result = result + term;
  }
  return result;
}

float surrogate_gradient_sigmoid(float x) {
  // Sigmoid: 1 / (1 + exp(-x))
  // Gradient: sigmoid(x) * (1 - sigmoid(x))
  float exp_neg_x = fast_exp(-x);
  float sigmoid_x = 1.0f / (1.0f + exp_neg_x);
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

float fast_tanh_derivative(float x) {
  float tanh_x = fast_tanh(x);
  return 1.0f - tanh_x * tanh_x;
}

void matrix_multiply(const int *weights, const float *input, float *output,
                     int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
      sum += FIXED_TO_FLOAT(weights[i * cols + j]) * input[j];
    }
    output[i] = sum;
  }
}

// LIF neuron structure
typedef struct {
  int v_membrane;
  int v_threshold;
  int tau_membrane;
  float voltage_trace;
  float spike_trace;
} lif_neuron_t;

void init_lif_neuron(lif_neuron_t *neuron) {
  neuron->v_membrane = V_RESET;
  neuron->v_threshold = V_THRESHOLD;
  neuron->tau_membrane = TAU_MEMBRANE;
  neuron->spike_trace = 0.0f;
  neuron->voltage_trace = 0.0f;
}

int update_lif_neuron_with_gradient(lif_neuron_t *neuron, int current_input,
                                    float *gradient) {
  float decay = FIXED_TO_FLOAT(DT) / FIXED_TO_FLOAT(neuron->tau_membrane);
  float new_voltage =
      FIXED_TO_FLOAT(neuron->v_membrane) * (1.0f - decay) + current_input;
  neuron->voltage_trace = new_voltage;

  if (new_voltage >= FIXED_TO_FLOAT(neuron->v_threshold)) {
    neuron->v_membrane = V_RESET;
    *gradient = surrogate_gradient_sigmoid(new_voltage);
    neuron->spike_trace = 1.0f;
    return 1;
  } else {
    neuron->v_membrane = FLOAT_TO_FIXED(new_voltage);
    *gradient = surrogate_gradient_sigmoid(new_voltage);
    neuron->spike_trace = 0.0f;
    return 0;
  }
}

// SNN forward pass with surrogate gradients
void snn_forward_pass(const float *state, float *action, float *log_prob) {
  static lif_neuron_t lif1_neurons[HIDDEN1_SIZE];
  static lif_neuron_t lif2_neurons[HIDDEN2_SIZE];

  int hidden1_input[HIDDEN1_SIZE];
  int hidden2_input[HIDDEN2_SIZE];
  int mu_output[NUM_ACTIONS];
  int logstd_output[NUM_ACTIONS];

  // Initialize neurons
  for (int i = 0; i < HIDDEN1_SIZE; i++) {
    init_lif_neuron(&lif1_neurons[i]);
  }
  for (int i = 0; i < HIDDEN2_SIZE; i++) {
    init_lif_neuron(&lif2_neurons[i]);
  }

  // Simulate SNN over time steps
  for (int t = 0; t < TIME_STEPS; t++) {
    // Layer 1: state -> hidden1
    float temp_hidden1[HIDDEN1_SIZE]; // Use float array instead of int
    matrix_multiply((int *)fc1_weights, state, temp_hidden1, HIDDEN1_SIZE,
                    NUM_STATES);
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
      temp_hidden1[i] += FIXED_TO_FLOAT(fc1_bias[i]);
      float grad;
      update_lif_neuron_with_gradient(&lif1_neurons[i], temp_hidden1[i], &grad);
      hidden1_trace[t][i] = FIXED_TO_FLOAT(lif1_neurons[i].v_membrane);
      lif1_spikes[t][i] = lif1_neurons[i].spike_trace;
    }

    // Layer 2: hidden1 -> hidden2
    float lif1_outputs[HIDDEN1_SIZE];
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
      lif1_outputs[i] = FIXED_TO_FLOAT(lif1_neurons[i].v_membrane);
    }
    float temp_hidden2[HIDDEN2_SIZE]; // Use float array
    matrix_multiply((int *)fc2_weights, lif1_outputs, temp_hidden2,
                    HIDDEN2_SIZE, HIDDEN1_SIZE);
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
      temp_hidden2[i] += FIXED_TO_FLOAT(fc2_bias[i]);
      float grad;
      update_lif_neuron_with_gradient(&lif2_neurons[i], temp_hidden2[i], &grad);
      hidden2_trace[t][i] = FIXED_TO_FLOAT(lif2_neurons[i].v_membrane);
      lif2_spikes[t][i] = lif2_neurons[i].spike_trace;
    }

    // Output layers
    float lif2_outputs[HIDDEN2_SIZE];
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
      lif2_outputs[i] = FIXED_TO_FLOAT(lif2_neurons[i].v_membrane);
    }
    float temp_mu[NUM_ACTIONS];
    matrix_multiply((int *)mu_weights, lif2_outputs, temp_mu, NUM_ACTIONS,
                    HIDDEN2_SIZE);
    for (int i = 0; i < NUM_ACTIONS; i++) {
      temp_mu[i] += FIXED_TO_FLOAT(mu_bias[i]);
    }
    float temp_logstd[NUM_ACTIONS];
    matrix_multiply((int *)logstd_weights, lif2_outputs, temp_logstd,
                    NUM_ACTIONS, HIDDEN2_SIZE);
    for (int i = 0; i < NUM_ACTIONS; i++) {
      temp_logstd[i] += FIXED_TO_FLOAT(logstd_bias[i]);
      if (temp_logstd[i] < -20.0f)
        temp_logstd[i] = -20.0f;
      if (temp_logstd[i] > 2.0f)
        temp_logstd[i] = 2.0f;
    }
    for (int i = 0; i < NUM_ACTIONS; i++) {
      action[i] = fast_tanh(temp_mu[i]);
      log_prob[i] = temp_logstd[i];
    }
  }
  void snn_backward_pass(const float *state, const float *action,
                         const float *critic_gradient) {
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
      for (int j = 0; j < NUM_STATES; j++) {
        fc1_grad[i][j] = 0;
      }
      fc1_bias[i] = 0;
    }
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
      for (int j = 0; j < HIDDEN1_SIZE; j++) {
        fc2_grad[i][j] = 0;
      }
      fc2_bias[i] = 0;
    }
    for (int i = 0; i < NUM_ACTIONS; i++) {
      for (int j = 0; j < HIDDEN2_SIZE; j++) {
        mu_grad[i][j] = 0;
        logstd_grad[i][j] = 0;
      }
      mu_bias[i] = 0;
      logstd_bias[i] = 0;
    }
    float grad_mu[NUM_ACTIONS];
    float grad_logstd[NUM_ACTIONS];
    for (int i = 0; i < NUM_ACTIONS; i++) {
      grad_mu[i] =
          critic_gradient[i] * fast_tanh_derivative(fast_tanh(action[i]));
      grad_logstd[i] = 0.0f;
    }
    for (int i = 0; i < NUM_ACTIONS; i++) {
      for (int j = 0; j < HIDDEN2_SIZE; j++) {
        float avg_lif2_activity = 0.0f;
        for (int t = 0; t < TIME_STEPS; t++) {
          avg_lif2_activity += hidden2_trace[t][j];
        }
        avg_lif2_activity /= TIME_STEPS;
        mu_grad[i][j] = FLOAT_TO_FIXED(grad_mu[i] * avg_lif2_activity);
      }
      mu_bias[i] = FLOAT_TO_FIXED(grad_mu[i]);
    }
    float grad_lif2[HIDDEN2_SIZE];
    for (int j = 0; j < HIDDEN2_SIZE; j++) {
      grad_lif2[j] = 0.0f;
      for (int i = 0; i < NUM_ACTIONS; i++) {
        grad_lif2[j] += grad_mu[i] * FIXED_TO_FLOAT(mu_weights[i][j]);
      }
    }
    for (int t = TIME_STEPS - 1; t >= 0; t--) {
      for (int i = 0; i < HIDDEN2_SIZE; i++) {
        float surrogate_grad = surrogate_gradient_sigmoid(hidden2_trace[t][i]);
        float grad = grad_lif2[i] * surrogate_grad;

        // Accumulate gradients for LIF2 weights
        float avg_lif1_activity = 0.0f;
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
          avg_lif1_activity += hidden1_trace[t][j];
        }
        avg_lif1_activity /= HIDDEN1_SIZE;

        for (int j = 0; j < HIDDEN1_SIZE; j++) {
          fc2_grad[i][j] += FLOAT_TO_FIXED(grad * hidden1_trace[t][j]);
        }
        fc2_bias[i] += FLOAT_TO_FIXED(grad);
      }
    }
    float grad_lif1[HIDDEN1_SIZE];
    for (int j = 0; j < HIDDEN1_SIZE; j++) {
      grad_lif1[j] = 0.0f;
      for (int i = 0; i < HIDDEN2_SIZE; i++) {
        for (int t = 0; t < TIME_STEPS; t++) {
          grad_lif1[j] += FIXED_TO_FLOAT(fc2_weights[i][j]) *
                          surrogate_gradient_sigmoid(hidden2_trace[t][i]) *
                          grad_lif2[i];
        }
      }
    }
    for (int t = TIME_STEPS - 1; t >= 0; t--) {
      for (int i = 0; i < HIDDEN1_SIZE; i++) {
        float surrogate_grad = surrogate_gradient_sigmoid(hidden1_trace[t][i]);
        float grad = grad_lif1[i] * surrogate_grad;

        for (int j = 0; j < NUM_STATES; j++) {
          fc1_grad[i][j] += FLOAT_TO_FIXED(grad * state[j]);
        }
        fc1_bias[i] += FLOAT_TO_FIXED(grad);
      }
    }
  }
}
void update_weights() {
  float learning_rate = FIXED_TO_FLOAT(LEARNING_RATE);

  // Update FC1 weights
  for (int i = 0; i < HIDDEN1_SIZE; i++) {
    for (int j = 0; j < NUM_STATES; j++) {
      int grad_update =
          FLOAT_TO_FIXED(learning_rate * FIXED_TO_FLOAT(fc1_grad[i][j]));
      fc1_weights[i][j] -= grad_update;
    }
    fc1_bias[i] -= FLOAT_TO_FIXED(learning_rate * FIXED_TO_FLOAT(fc1_bias[i]));
  }

  // Update FC2 weights
  for (int i = 0; i < HIDDEN2_SIZE; i++) {
    for (int j = 0; j < HIDDEN1_SIZE; j++) {
      int grad_update =
          FLOAT_TO_FIXED(learning_rate * FIXED_TO_FLOAT(fc2_grad[i][j]));
      fc2_weights[i][j] -= grad_update;
    }
    fc2_bias[i] -= FLOAT_TO_FIXED(learning_rate * FIXED_TO_FLOAT(fc2_bias[i]));
  }

  // Update output weights
  for (int i = 0; i < NUM_ACTIONS; i++) {
    for (int j = 0; j < HIDDEN2_SIZE; j++) {
      int grad_update =
          FLOAT_TO_FIXED(learning_rate * FIXED_TO_FLOAT(mu_grad[i][j]));
      mu_weights[i][j] -= grad_update;

      grad_update =
          FLOAT_TO_FIXED(learning_rate * FIXED_TO_FLOAT(logstd_grad[i][j]));
      logstd_weights[i][j] -= grad_update;
    }
    mu_bias[i] -= FLOAT_TO_FIXED(learning_rate * FIXED_TO_FLOAT(mu_bias[i]));
    logstd_bias[i] -=
        FLOAT_TO_FIXED(learning_rate * FIXED_TO_FLOAT(logstd_bias[i]));
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
          FLOAT_TO_FIXED((float)simple_rand() / RAND_MAX * 0.2f - 0.1f);
    }
    fc1_bias[i] = 0;
  }

  for (int i = 0; i < HIDDEN2_SIZE; i++) {
    for (int j = 0; j < HIDDEN1_SIZE; j++) {
      fc2_weights[i][j] =
          FLOAT_TO_FIXED((float)simple_rand() / RAND_MAX * 0.2f - 0.1f);
    }
    fc2_bias[i] = 0;
  }

  for (int i = 0; i < NUM_ACTIONS; i++) {
    for (int j = 0; j < HIDDEN2_SIZE; j++) {
      mu_weights[i][j] =
          FLOAT_TO_FIXED((float)simple_rand() / RAND_MAX * 0.2f - 0.1f);
      logstd_weights[i][j] =
          FLOAT_TO_FIXED((float)simple_rand() / RAND_MAX * 0.2f - 0.1f);
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

    // Forward pass through SNN
    float action[NUM_ACTIONS];
    float log_prob[NUM_ACTIONS]; // Changed to array

    snn_forward_pass(state, action, log_prob);

    // Send action and log_prob back to PC
    char response[sizeof(float) * (NUM_ACTIONS + 1)];
    memcpy(response, action, sizeof(float) * NUM_ACTIONS);
    memcpy(response + sizeof(float) * NUM_ACTIONS, log_prob, sizeof(float));

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
