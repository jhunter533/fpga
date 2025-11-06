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
#define NUM_STATES 3
#define HIDDEN1_SIZE 128
#define HIDDEN2_SIZE 64
#define NUM_ACTIONS 1
#define TIME_STEPS 5

// Fixed-point arithmetic
#define FIXED_POINT_SCALE 65536.0f
#define FIXED_TO_FLOAT(x) ((float)(x) / FIXED_POINT_SCALE)
#define FLOAT_TO_FIXED(x) ((int)((x) * FIXED_POINT_SCALE))

// LIF Neuron parameters
#define V_THRESHOLD 1.0f // Fixed point: 1.0
#define V_RESET 0
#define TAU_MEMBRANE 20.0f // Fixed point: 20.0
#define DT 1.0f            // Fixed point: 1.0

// Training parameters
#define LEARNING_RATE .001f // Fixed point: 0.3

// Network weights
static int fc1_weights[HIDDEN1_SIZE][NUM_STATES];
static int fc2_weights[HIDDEN2_SIZE][HIDDEN1_SIZE];
static int mu_weights[NUM_ACTIONS][HIDDEN2_SIZE];
static int logstd_weights[NUM_ACTIONS][HIDDEN2_SIZE];

static int fc1_bias[HIDDEN1_SIZE];
static int fc2_bias[HIDDEN2_SIZE];
static int mu_bias[NUM_ACTIONS];
static int logstd_bias[NUM_ACTIONS];

// Gradient matrices
static float fc1_grad[HIDDEN1_SIZE][NUM_STATES];
static float fc2_grad[HIDDEN2_SIZE][HIDDEN1_SIZE];
static float mu_grad[NUM_ACTIONS][HIDDEN2_SIZE];
static float logstd_grad[NUM_ACTIONS][HIDDEN2_SIZE];

static float fc1_bias_grad[HIDDEN1_SIZE];
static float fc2_bias_grad[HIDDEN2_SIZE];
static float mu_bias_grad[NUM_ACTIONS];
static float logstd_bias_grad[NUM_ACTIONS];

// LIF neuron structure
typedef struct {
  float v_membrane;
  float v_threshold;
  float tau_membrane;
  float spike_trace;
  float voltage_trace;
} lif_neuron_t;

// LI neuron
typedef struct {
  float v_membrane;
  float tau_membrane;
  float leak_rate;
} li_neuron_t;

static float hidden1_voltages[TIME_STEPS][HIDDEN1_SIZE];
static float hidden1_spikes[TIME_STEPS][HIDDEN1_SIZE];
static float hidden2_voltages[TIME_STEPS][HIDDEN2_SIZE];
static float hidden2_spikes[TIME_STEPS][HIDDEN2_SIZE];

static float hidden1_inputs[TIME_STEPS][HIDDEN1_SIZE];
static float hidden2_inputs[TIME_STEPS][HIDDEN2_SIZE];

// Initialize random number generator
static unsigned int simple_rand_state = 12345;

int simple_rand() {
  simple_rand_state = simple_rand_state * 1103515245 + 12345;
  return (unsigned int)(simple_rand_state / 65536) % 32768;
}

// Surrogate gradient function
float surrogate_gradient(float membrane_voltage, float threshold) {
  float x = membrane_voltage - threshold;
  float k = 0.5f; // Steepness parameter
  return k * expf(-fabsf(x));
}

// Fast tanh approximation
float fast_tanh(float x) {
  if (x > 3.0f)
    return 1.0f;
  if (x < -3.0f)
    return -1.0f;
  float x2 = x * x;
  return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

float fast_tanh_derivative(float x) {
  float tanh_x = fast_tanh(x);
  return 1.0f - tanh_x * tanh_x;
}

// LIF neuron update
int update_lif_neuron(lif_neuron_t *neuron, float input) {
  float decay = expf(-DT / neuron->tau_membrane);
  neuron->voltage_trace = neuron->v_membrane * decay + input;

  if (neuron->voltage_trace >= neuron->threshold) {
    neuron->spike_trace = 1.0f;
    neuron->v_membrane = 0.0f; // Reset
    return 1;                  // Spike
  } else {
    neuron->spike_trace = 0.0f;
    neuron->v_membrane = neuron->voltage_trace;
    return 0; // No spike
  }
}

// LI neuron update (for continuous output)
float update_li_neuron(li_neuron_t *neuron, float input) {
  float decay = expf(-DT / neuron->tau_membrane);
  neuron->v_membrane = neuron->v_membrane * decay + input;
  return neuron->v_membrane;
}

void matrix_multiply(const int *weights, const float *input, float *output,
                     int rows, int cols) {
  // Use loop unrolling to reduce loop overhead
  for (int i = 0; i < rows; i++) {
    float sum = 0.0f;

    // Process 4 elements at a time for better cache usage
    int j = 0;
    for (; j < cols - 3; j += 4) {
      sum += weights[i * cols + j] * input[j] +
             weights[i * cols + j + 1] * input[j + 1] +
             weights[i * cols + j + 2] * input[j + 2] +
             weights[i * cols + j + 3] * input[j + 3];
    }

    // Handle remaining elements
    for (; j < cols; j++) {
      sum += weights[i * cols + j] * input[j];
    }

    output[i] = sum;
  }
}

// SNN Forward Pass with trace recording
void snn_forward_pass(const float *state, float *action, float *log_prob) {
  static lif_neuron_t lif1_neurons[HIDDEN1_SIZE];
  static lif_neuron_t lif2_neurons[HIDDEN2_SIZE];
  static li_neuron_t mu_neurons[NUM_ACTIONS];
  static li_neuron_t logstd_neurons[NUM_ACTIONS];

  float hidden1[HIDDEN1_SIZE];
  float hidden2[HIDDEN2_SIZE];
  float mu_output[NUM_ACTIONS];
  float logstd_output[NUM_ACTIONS];

  // Initialize neurons
  for (int i = 0; i < HIDDEN1_SIZE; i++) {
    lif1_neurons[i].v_membrane = 0.0f;
    lif1_neurons[i].voltage_trace = 0.0f;
    lif1_neurons[i].spike_trace = 0.0f;
    lif1_neurons[i].threshold = V_THRESHOLD;
    lif1_neurons[i].tau_membrane = TAU_MEMBRANE;
  }
  for (int i = 0; i < HIDDEN2_SIZE; i++) {
    lif2_neurons[i].v_membrane = 0.0f;
    lif2_neurons[i].voltage_trace = 0.0f;
    lif2_neurons[i].spike_trace = 0.0f;
    lif2_neurons[i].threshold = V_THRESHOLD;
    lif2_neurons[i].tau_membrane = TAU_MEMBRANE;
  }
  for (int i = 0; i < NUM_ACTIONS; i++) {
    mu_neurons[i].v_membrane = 0.0f;
    mu_neurons[i].tau_membrane = 20.0f;
    mu_neurons[i].leak_rate = 0.1f;
    logstd_neurons[i].v_membrane = 0.0f;
    logstd_neurons[i].tau_membrane = 20.0f;
    logstd_neurons[i].leak_rate = 0.1f;
  }

  // Time-stepped simulation
  for (int t = 0; t < TIME_STEPS; t++) {
    // Layer 1: state -> hidden1
    matrix_multiply((float *)fc1_weights, state, hidden1, HIDDEN1_SIZE,
                    NUM_STATES);
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
      float input = hidden1[i] + fc1_bias[i];
      hidden1_inputs[t][i] = input;
      update_lif_neuron(&lif1_neurons[i], input);

      // Store traces
      hidden1_voltages[t][i] = lif1_neurons[i].voltage_trace;
      hidden1_spikes[t][i] = lif1_neurons[i].spike_trace;
    }

    // Layer 2: hidden1 -> hidden2
    matrix_multiply((float *)fc2_weights, hidden1, hidden2, HIDDEN2_SIZE,
                    HIDDEN1_SIZE);
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
      float input = hidden2[i] + fc2_bias[i];
      hidden2_inputs[t][i] = input;
      update_lif_neuron(&lif2_neurons[i], input);

      // Store traces
      hidden2_voltages[t][i] = lif2_neurons[i].voltage_trace;
      hidden2_spikes[t][i] = lif2_neurons[i].spike_trace;
    }
  }

  // Output layers: LI neurons for continuous output
  matrix_multiply((float *)mu_weights, hidden2, mu_output, NUM_ACTIONS,
                  HIDDEN2_SIZE);
  matrix_multiply((float *)logstd_weights, hidden2, logstd_output, NUM_ACTIONS,
                  HIDDEN2_SIZE);

  for (int i = 0; i < NUM_ACTIONS; i++) {
    float mu_val = update_li_neuron(&mu_neurons[i], mu_output[i] + mu_bias[i]);
    float logstd_val =
        update_li_neuron(&logstd_neurons[i], logstd_output[i] + logstd_bias[i]);

    action[i] = fast_tanh(mu_val);
    log_prob[i] =
        logstd_val > 2.0f ? 2.0f : (logstd_val < -20.0f ? -20.0f : logstd_val);
  }
}
void snn_backward_pass(const float *state, const float *action_grad,
                       const float *logprob_grad) {
  // Initialize gradients to zero
  for (int i = 0; i < HIDDEN1_SIZE; i++) {
    for (int j = 0; j < NUM_STATES; j++) {
      fc1_grad[i][j] = 0.0f;
    }
    fc1_bias_grad[i] = 0.0f;
  }
  for (int i = 0; i < HIDDEN2_SIZE; i++) {
    for (int j = 0; j < HIDDEN1_SIZE; j++) {
      fc2_grad[i][j] = 0.0f;
    }
    fc2_bias_grad[i] = 0.0f;
  }
  for (int i = 0; i < NUM_ACTIONS; i++) {
    for (int j = 0; j < HIDDEN2_SIZE; j++) {
      mu_grad[i][j] = 0.0f;
      logstd_grad[i][j] = 0.0f;
    }
    mu_bias_grad[i] = 0.0f;
    logstd_bias_grad[i] = 0.0f;
  }

  // Calculate gradients for output layer
  for (int i = 0; i < NUM_ACTIONS; i++) {
    // Derivative of tanh for action
    float mu_grad_output = action_grad[i] * fast_tanh_derivative(action[i]);
    float logstd_grad_output = logprob_grad[i];

    // Backpropagate to output weights
    for (int j = 0; j < HIDDEN2_SIZE; j++) {
      // Average activity over time for gradient calculation
      float avg_hidden2_activity = 0.0f;
      for (int t = 0; t < TIME_STEPS; t++) {
        avg_hidden2_activity += hidden2_voltages[t][j];
      }
      avg_hidden2_activity /= TIME_STEPS;

      mu_grad[i][j] += mu_grad_output * avg_hidden2_activity;
      logstd_grad[i][j] += logstd_grad_output * avg_hidden2_activity;
    }
    mu_bias_grad[i] += mu_grad_output;
    logstd_bias_grad[i] += logstd_grad_output;
  }

  // Backpropagate to hidden layer 2
  float grad_hidden2[HIDDEN2_SIZE];
  for (int j = 0; j < HIDDEN2_SIZE; j++) {
    grad_hidden2[j] = 0.0f;
    for (int i = 0; i < NUM_ACTIONS; i++) {
      grad_hidden2[j] += (mu_grad_output * mu_weights[i][j] +
                          logstd_grad_output * logstd_weights[i][j]);
    }
  }

  // Backpropagate through time for LIF neurons
  for (int t = TIME_STEPS - 1; t >= 0; t--) {
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
      // Surrogate gradient for LIF neuron
      float surrogate_grad =
          surrogate_gradient(hidden2_voltages[t][i], V_THRESHOLD);
      float total_grad = grad_hidden2[i] * surrogate_grad;

      // Accumulate gradients for hidden2 weights
      float input_activity = hidden2_inputs[t][i];
      for (int j = 0; j < HIDDEN1_SIZE; j++) {
        fc2_grad[i][j] += total_grad * hidden1_voltages[t][j];
      }
      fc2_bias_grad[i] += total_grad;
    }
  }

  // Backpropagate to hidden layer 1
  float grad_hidden1[HIDDEN1_SIZE];
  for (int j = 0; j < HIDDEN1_SIZE; j++) {
    grad_hidden1[j] = 0.0f;
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
      grad_hidden1[j] += grad_hidden2[i] * fc2_weights[i][j];
    }
  }

  // Backpropagate through time for first hidden layer
  for (int t = TIME_STEPS - 1; t >= 0; t--) {
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
      // Surrogate gradient for LIF neuron
      float surrogate_grad =
          surrogate_gradient(hidden1_voltages[t][i], V_THRESHOLD);
      float total_grad = grad_hidden1[i] * surrogate_grad;

      // Accumulate gradients for hidden1 weights
      for (int j = 0; j < NUM_STATES; j++) {
        fc1_grad[i][j] += total_grad * state[j];
      }
      fc1_bias_grad[i] += total_grad;
    }
  }

  // Update weights using accumulated gradients
  for (int i = 0; i < HIDDEN1_SIZE; i++) {
    for (int j = 0; j < NUM_STATES; j++) {
      fc1_weights[i][j] -= LEARNING_RATE * fc1_grad[i][j];
    }
    fc1_bias[i] -= LEARNING_RATE * fc1_bias_grad[i];
  }
  for (int i = 0; i < HIDDEN2_SIZE; i++) {
    for (int j = 0; j < HIDDEN1_SIZE; j++) {
      fc2_weights[i][j] -= LEARNING_RATE * fc2_grad[i][j];
    }
    fc2_bias[i] -= LEARNING_RATE * fc2_bias_grad[i];
  }
  for (int i = 0; i < NUM_ACTIONS; i++) {
    for (int j = 0; j < HIDDEN2_SIZE; j++) {
      mu_weights[i][j] -= LEARNING_RATE * mu_grad[i][j];
      logstd_weights[i][j] -= LEARNING_RATE * logstd_grad[i][j];
    }
    mu_bias[i] -= LEARNING_RATE * mu_bias_grad[i];
    logstd_bias[i] -= LEARNING_RATE * logstd_bias_grad[i];
  }
}

// Initialize weights with proper distribution
void initialize_weights() {
  for (int i = 0; i < HIDDEN1_SIZE; i++) {
    for (int j = 0; j < NUM_STATES; j++) {
      float rand_val = ((float)simple_rand() / 32767.0f - 0.5f) * 0.4f;
      fc1_weights[i][j] = rand_val;
    }
    fc1_bias[i] = 0.0f;
  }

  for (int i = 0; i < HIDDEN2_SIZE; i++) {
    for (int j = 0; j < HIDDEN1_SIZE; j++) {
      float rand_val = ((float)simple_rand() / 32767.0f - 0.5f) * 0.4f;
      fc2_weights[i][j] = rand_val;
    }
    fc2_bias[i] = 0.0f;
  }

  for (int i = 0; i < NUM_ACTIONS; i++) {
    for (int j = 0; j < HIDDEN2_SIZE; j++) {
      float rand_val = ((float)simple_rand() / 32767.0f - 0.5f) * 0.2f;
      mu_weights[i][j] = rand_val;
      logstd_weights[i][j] = rand_val;
    }
    mu_bias[i] = 0.0f;
    logstd_bias[i] = 0.0f;
  }
}

// TCP data handling
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
    float log_prob[NUM_ACTIONS];

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
