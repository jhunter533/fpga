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

struct netif *app_netif;
static struct netif server_netif;
struct netif *echo_netif;

static unsigned int simple_rand_state = 12345u;

int simple_rand() {
  simple_rand_state = simple_rand_state * 1103515245u + 12345u;
  return (unsigned int)(simple_rand_state / 65536) % 32768;
}

void tcp_fasttmr(void);
void tcp_slowtmr(void);

#define TCP_PORT 12345
#define NUM_STATES 3
#define HIDDEN1_SIZE 64
#define HIDDEN2_SIZE 32
#define NUM_ACTIONS 1
#define TIME_STEPS 5

// LIF Neuron parameters
#define V_THRESHOLD 1.0f // Fixed point: 1.0
#define V_RESET 0
#define TAU_MEMBRANE 20.0f // Fixed point: 20.0
#define DT 1.0f            // Fixed point: 1.0

// Training parameters
#define LEARNING_RATE .003f // Fixed point: 0.3

// HEADER DEFINITION
#pragma pack(push, 1)
#define MSG_MAGIC 0xDEADBEEF

typedef struct {
  uint32_t magic;
  uint16_t version;
  uint16_t msg_type;
  uint32_t seq_no;
  uint32_t payload_len;
} msg_header_t;

typedef enum {
  MSG_TYPE_ACTOR_QUERY = 1,
  MSG_TYPE_ACTOR_RESPONSE = 2,
  MSG_TYPE_MINIBATCH_QUERY = 3,
  MSG_TYPE_MINIBATCH_RESP = 4,
  MSG_TYPE_GRAD_UPDATE = 5,
  MSG_TYPE_ACK = 6,
  MSG_TYPE_PING = 7,
  MSG_TYPE_PONG = 8,
} msg_type_t;

#pragma pack(pop)

// Network weights
static float fc1_weights[HIDDEN1_SIZE][NUM_STATES];
static float fc2_weights[HIDDEN2_SIZE][HIDDEN1_SIZE];
static float mu_weights[NUM_ACTIONS][HIDDEN2_SIZE];
static float logstd_weights[NUM_ACTIONS][HIDDEN2_SIZE];

static float fc1_bias[HIDDEN1_SIZE];
static float fc2_bias[HIDDEN2_SIZE];
static float mu_bias[NUM_ACTIONS];
static float logstd_bias[NUM_ACTIONS];

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
} li_neuron_t;

static float hidden1_voltages[TIME_STEPS][HIDDEN1_SIZE];
static float hidden1_spikes[TIME_STEPS][HIDDEN1_SIZE];
static float hidden2_voltages[TIME_STEPS][HIDDEN2_SIZE];
static float hidden2_spikes[TIME_STEPS][HIDDEN2_SIZE];

static float hidden1_inputs[TIME_STEPS][HIDDEN1_SIZE];
static float hidden2_inputs[TIME_STEPS][HIDDEN2_SIZE];

static float stored_action[NUM_ACTIONS];
static float stored_log_prob[NUM_ACTIONS];

static float stored_mu_grad_output[NUM_ACTIONS];
static float stored_logstd_grad_output[NUM_ACTIONS];

typedef struct {
  float state[NUM_STATES];
  float action[NUM_ACTIONS];
  float log_prob;
  float hidden1_v[TIME_STEPS][HIDDEN1_SIZE];
  float hidden2_v[TIME_STEPS][HIDDEN2_SIZE];
  float hidden1_in[TIME_STEPS][HIDDEN1_SIZE];
  float hidden2_in[TIME_STEPS][HIDDEN2_SIZE];
} sample_trace_t;

#define MAX_BATCH_SIZE 256
static sample_trace_t batch_traces[MAX_BATCH_SIZE];
static int batch_size = 0;
static volatile uint32_t global_seq_no = 0;

float fast_exp_neg(float x) {
  if (x > 10.0f)
    return 0.000045f; // e^(-10)
  if (x < -10.0f)
    return 22026.47f; // e^10
  // Taylor series: e^x = 1 + x + x^2/2! + x^3/3! + ...
  float result = 1.0f;
  float term = 1.0f;
  for (int i = 1; i <= 8; i++) {
    term *= x / i;
    result += term;
  }
  return result;
}

// Fast absolute value
float fast_fabs(float x) { return x > 0 ? x : -x; }

// Surrogate gradient function
float surrogate_gradient(float membrane_voltage, float threshold) {
  float x = membrane_voltage - threshold;
  float k = 0.5f; // Steepness parameter
  return k * fast_exp_neg(-fast_fabs(x));
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
  const float decay = .9512;
  neuron->voltage_trace = neuron->v_membrane * decay + input;

  if (neuron->voltage_trace >= neuron->v_threshold) {
    neuron->spike_trace = 1.0f;
    neuron->v_membrane = V_RESET; // Reset
    return 1;                     // Spike
  } else {
    neuron->spike_trace = 0.0f;
    neuron->v_membrane = neuron->voltage_trace;
    return 0; // No spike
  }
}

// LI neuron update (for continuous output)
float update_li_neuron(li_neuron_t *neuron, float input) {
  float decay = .9512f;
  neuron->v_membrane = neuron->v_membrane * decay + input;
  return neuron->v_membrane;
}

void matrix_multiply(const float *weights, const float *input, float *output,
                     int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
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
    lif1_neurons[i].v_threshold = V_THRESHOLD;
    lif1_neurons[i].tau_membrane = TAU_MEMBRANE;
    lif1_neurons[i].spike_trace = 0.0f;
    lif1_neurons[i].voltage_trace = 0.0f;
  }
  for (int i = 0; i < HIDDEN2_SIZE; i++) {
    lif2_neurons[i].v_membrane = 0.0f;
    lif2_neurons[i].v_threshold = V_THRESHOLD;
    lif2_neurons[i].tau_membrane = TAU_MEMBRANE;
    lif2_neurons[i].spike_trace = 0.0f;
    lif2_neurons[i].voltage_trace = 0.0f;
  }
  for (int i = 0; i < NUM_ACTIONS; i++) {
    mu_neurons[i].v_membrane = 0.0f;
    mu_neurons[i].tau_membrane = TAU_MEMBRANE;
    logstd_neurons[i].v_membrane = 0.0f;
    logstd_neurons[i].tau_membrane = TAU_MEMBRANE;
  }

  // Time-stepped simulation
  for (int t = 0; t < TIME_STEPS; t++) {
    // Layer 1: state -> hidden1 (LIF)
    matrix_multiply((float *)fc1_weights, state, hidden1, HIDDEN1_SIZE,
                    NUM_STATES);
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
      float input = hidden1[i] + fc1_bias[i];
      hidden1_inputs[t][i] = input;
      update_lif_neuron(&lif1_neurons[i], input);
      hidden1_voltages[t][i] = lif1_neurons[i].voltage_trace;
      hidden1_spikes[t][i] = lif1_neurons[i].spike_trace;
    }

    // Layer 2: hidden1 -> hidden2 (LIF)
    matrix_multiply((float *)fc2_weights, hidden1, hidden2, HIDDEN2_SIZE,
                    HIDDEN1_SIZE);
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
      float input = hidden2[i] + fc2_bias[i];
      hidden2_inputs[t][i] = input;
      update_lif_neuron(&lif2_neurons[i], input);
      hidden2_voltages[t][i] = lif2_neurons[i].voltage_trace;
      hidden2_spikes[t][i] = lif2_neurons[i].spike_trace;
    }
  }

  // Output layers: LI neurons
  matrix_multiply((float *)mu_weights, hidden2, mu_output, NUM_ACTIONS,
                  HIDDEN2_SIZE);
  matrix_multiply((float *)logstd_weights, hidden2, logstd_output, NUM_ACTIONS,
                  HIDDEN2_SIZE);

  for (int i = 0; i < NUM_ACTIONS; i++) {
    float mu_val = update_li_neuron(&mu_neurons[i], mu_output[i] + mu_bias[i]);
    float logstd_val =
        update_li_neuron(&logstd_neurons[i], logstd_output[i] + logstd_bias[i]);

    action[i] = fast_tanh(mu_val);
    log_prob[i] = (logstd_val > 2.0f)
                      ? 2.0f
                      : ((logstd_val < -20.0f) ? -20.0f : logstd_val);
  }

  for (int i = 0; i < NUM_ACTIONS; i++) {
    stored_action[i] = action[i];
    stored_log_prob[i] = log_prob[i];
  }
}

void snn_backward_pass(const sample_trace_t *trace, const float *dL_da,
                       const float *dL_dlogp) {

  // 1. Output layer gradients
  for (int i = 0; i < NUM_ACTIONS; i++) {
    stored_mu_grad_output[i] =
        dL_da[i] * fast_tanh_derivative(trace->action[i]);
    stored_logstd_grad_output[i] = dL_dlogp[i];
  }

  // 2. Backprop to hidden2
  float grad_hidden2[HIDDEN2_SIZE] = {0};
  for (int j = 0; j < HIDDEN2_SIZE; j++) {
    for (int i = 0; i < NUM_ACTIONS; i++) {
      grad_hidden2[j] += (stored_mu_grad_output[i] * mu_weights[i][j] +
                          stored_logstd_grad_output[i] * logstd_weights[i][j]);
    }
  }

  // 3. Backprop through time (LIF2)
  for (int t = TIME_STEPS - 1; t >= 0; t--) {
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
      float sg = surrogate_gradient(trace->hidden2_v[t][i], V_THRESHOLD);
      float total_grad = grad_hidden2[i] * sg;
      for (int j = 0; j < HIDDEN1_SIZE; j++) {
        fc2_grad[i][j] += total_grad * trace->hidden1_v[t][j];
      }
      fc2_bias_grad[i] += total_grad;
    }
  }

  // 4. Backprop to hidden1
  float grad_hidden1[HIDDEN1_SIZE] = {0};
  for (int j = 0; j < HIDDEN1_SIZE; j++) {
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
      grad_hidden1[j] += grad_hidden2[i] * fc2_weights[i][j];
    }
  }

  // 5. Backprop through time (LIF1)
  for (int t = TIME_STEPS - 1; t >= 0; t--) {
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
      float sg = surrogate_gradient(trace->hidden1_v[t][i], V_THRESHOLD);
      float total_grad = grad_hidden1[i] * sg;
      for (int j = 0; j < NUM_STATES; j++) {
        fc1_grad[i][j] += total_grad * trace->state[j];
      }
      fc1_bias_grad[i] += total_grad;
    }
  }

  // 6. Output weights (use last time step)
  for (int i = 0; i < NUM_ACTIONS; i++) {
    for (int j = 0; j < HIDDEN2_SIZE; j++) {
      mu_grad[i][j] +=
          stored_mu_grad_output[i] * trace->hidden2_v[TIME_STEPS - 1][j];
      logstd_grad[i][j] +=
          stored_logstd_grad_output[i] * trace->hidden2_v[TIME_STEPS - 1][j];
    }
    mu_bias_grad[i] += stored_mu_grad_output[i];
    logstd_bias_grad[i] += stored_logstd_grad_output[i];
  }
}

void apply_gradients() {
  unsigned int old_state = Xil_ExceptionGetStatus();
  Xil_ExceptionDisable();

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

  // Zero gradients
  memset(fc1_grad, 0, sizeof(fc1_grad));
  memset(fc2_grad, 0, sizeof(fc2_grad));
  memset(mu_grad, 0, sizeof(mu_grad));
  memset(logstd_grad, 0, sizeof(logstd_grad));
  memset(fc1_bias_grad, 0, sizeof(fc1_bias_grad));
  memset(fc2_bias_grad, 0, sizeof(fc2_bias_grad));
  memset(mu_bias_grad, 0, sizeof(mu_bias_grad));
  memset(logstd_bias_grad, 0, sizeof(logstd_bias_grad));

  Xil_ExceptionRestore(old_state);
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

err_t send_response(struct tcp_pcb *tpcb, uint16_t msg_type,
                    const void *payload, uint32_t len) {
  msg_header_t hdr = {.magic = MSG_MAGIC,
                      .version = 1,
                      .msg_type = msg_type,
                      .seq_no = ++global_seq_no,
                      .payload_len = len};

  uint32_t total_len = sizeof(hdr) + len;
  char *buf = mem_malloc(total_len);
  if (!buf)
    return ERR_MEM;

  memcpy(buf, &hdr, sizeof(hdr));
  if (len)
    memcpy(buf + sizeof(hdr), payload, len);

  err_t err = tcp_write(tpcb, buf, total_len, TCP_WRITE_FLAG_COPY);
  mem_free(buf);
  if (err == ERR_OK)
    tcp_output(tpcb);
  return err;
}

// ===== MESSAGE HANDLERS =====
err_t handle_actor_query(struct tcp_pcb *tpcb, const float *state, int done) {
  float action[NUM_ACTIONS];
  float log_prob;
  sample_trace_t trace;

  memcpy(trace.state, state, sizeof(trace.state));
  snn_forward_pass(state, action, &log_prob);

  // Save traces (using YOUR global arrays)
  for (int t = 0; t < TIME_STEPS; t++) {
    memcpy(trace.hidden1_v[t], hidden1_voltages[t],
           sizeof(hidden1_voltages[t]));
    memcpy(trace.hidden2_v[t], hidden2_voltages[t],
           sizeof(hidden2_voltages[t]));
    memcpy(trace.hidden1_in[t], hidden1_inputs[t], sizeof(hidden1_inputs[t]));
    memcpy(trace.hidden2_in[t], hidden2_inputs[t], sizeof(hidden2_inputs[t]));
  }
  memcpy(trace.action, action, sizeof(action));
  trace.log_prob = log_prob;

  batch_traces[0] = trace;
  batch_size = 1;

  char response[sizeof(float) * (NUM_ACTIONS + 1)];
  memcpy(response, action, sizeof(float) * NUM_ACTIONS);
  memcpy(response + sizeof(float) * NUM_ACTIONS, &log_prob, sizeof(float));

  return send_response(tpcb, MSG_TYPE_ACTOR_RESPONSE, response,
                       sizeof(response));
}

err_t handle_minibatch_query(struct tcp_pcb *tpcb, const uint8_t *payload,
                             uint32_t len) {
  if (len < 4)
    return ERR_ARG;
  uint32_t N = *(uint32_t *)payload;
  if (N == 0 || N > MAX_BATCH_SIZE)
    return ERR_ARG;

  uint32_t states_bytes = N * NUM_STATES * sizeof(float);
  if (4 + states_bytes != len)
    return ERR_ARG;

  const float *states = (const float *)(payload + 4);
  batch_size = N;

  uint32_t resp_size = 4 + N * (NUM_ACTIONS + 1) * sizeof(float);
  char *resp = mem_malloc(resp_size);
  if (!resp)
    return ERR_MEM;

  *(uint32_t *)resp = N;
  float *out = (float *)(resp + 4);

  for (uint32_t i = 0; i < N; i++) {
    const float *s = &states[i * NUM_STATES];
    sample_trace_t *trace = &batch_traces[i];

    memcpy(trace->state, s, sizeof(trace->state));
    snn_forward_pass(s, trace->action, &trace->log_prob);

    // Save traces
    for (int t = 0; t < TIME_STEPS; t++) {
      memcpy(trace->hidden1_v[t], hidden1_voltages[t],
             sizeof(hidden1_voltages[t]));
      memcpy(trace->hidden2_v[t], hidden2_voltages[t],
             sizeof(hidden2_voltages[t]));
      memcpy(trace->hidden1_in[t], hidden1_inputs[t],
             sizeof(hidden1_inputs[t]));
      memcpy(trace->hidden2_in[t], hidden2_inputs[t],
             sizeof(hidden2_inputs[t]));
    }

    memcpy(out, trace->action, sizeof(float) * NUM_ACTIONS);
    out[NUM_ACTIONS] = trace->log_prob;
    out += (NUM_ACTIONS + 1);
  }

  err_t err = send_response(tpcb, MSG_TYPE_MINIBATCH_RESP, resp, resp_size);
  mem_free(resp);
  return err;
}

err_t handle_grad_update(struct tcp_pcb *tpcb, const uint8_t *payload,
                         uint32_t len) {
  if (len < 4)
    return ERR_ARG;
  uint32_t N = *(uint32_t *)payload;
  if (N == 0 || N > MAX_BATCH_SIZE || N > batch_size)
    return ERR_ARG;

  uint32_t grad_bytes =
      N * (NUM_ACTIONS * 2) * sizeof(float); // dL/da + dL/dlogp
  if (4 + grad_bytes != len)
    return ERR_ARG;

  const float *grads = (const float *)(payload + 4);

  for (uint32_t i = 0; i < N; i++) {
    const float *dL_da = &grads[i * 2 * NUM_ACTIONS];
    const float *dL_dlogp = &grads[i * 2 * NUM_ACTIONS + NUM_ACTIONS];
    snn_backward_pass(&batch_traces[i], dL_da, dL_dlogp);
  }

  apply_gradients();
  return send_response(tpcb, MSG_TYPE_ACK, NULL, 0);
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

  if (p->tot_len >= sizeof(msg_header_t)) {
    msg_header_t hdr;
    memcpy(&hdr, p->payload, sizeof(hdr));

    if (hdr.magic == MSG_MAGIC && hdr.version == 1) {
      uint32_t total = sizeof(hdr) + hdr.payload_len;
      if (p->tot_len >= total) {
        const uint8_t *payload = (const uint8_t *)p->payload + sizeof(hdr);
        switch (hdr.msg_type) {
        case MSG_TYPE_ACTOR_QUERY:
          if (hdr.payload_len == (NUM_STATES + 1) * sizeof(float)) {
            float state[NUM_STATES];
            int done =
                (*(float *)(payload + NUM_STATES * sizeof(float)) > 0.5f);
            memcpy(state, payload, sizeof(state));
            handle_actor_query(tpcb, state, done);
          }
          break;
        case MSG_TYPE_MINIBATCH_QUERY:
          handle_minibatch_query(tpcb, payload, hdr.payload_len);
          break;
        case MSG_TYPE_GRAD_UPDATE:
          handle_grad_update(tpcb, payload, hdr.payload_len);
          break;
        case MSG_TYPE_PING:
          send_response(tpcb, MSG_TYPE_PONG, NULL, 0);
          break;
        default:
          xil_printf("Unknown msg type: %d\r\n", hdr.msg_type);
        }
      }
    } else {
      xil_printf("Bad magic/version: 0x%08X v%d\r\n", hdr.magic, hdr.version);
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
  struct tcp_pcb *pcb = tcp_new();
  if (pcb == NULL)
    return -1;

  err_t err = tcp_bind(pcb, IP_ADDR_ANY, TCP_PORT);
  if (err != ERR_OK) {
    tcp_close(pcb);
    return -1;
  }

  pcb = tcp_listen(pcb);
  if (pcb == NULL)
    return -1;

  tcp_accept(pcb, tcp_connection_accepted);
  xil_printf("TCP server listening on port %d\r\n", TCP_PORT);
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
