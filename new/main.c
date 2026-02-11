#include "lwip/init.h"
#include "lwip/netif.h"
#include "lwip/tcp.h"
#include "lwip/timeouts.h"
#include "netif/xadapter.h"
#include "pl_driver.h" // ← only include driver, not implementation
#include "platform.h"
#include "platform_config.h"
#include "xil_cache.h"
#include "xil_exception.h"
#include "xil_printf.h"
#include "xparameters.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

extern volatile int TcpFastTmrFlag;
extern volatile int TcpSlowTmrFlag;

struct netif *app_netif;
static struct netif server_netif;
#define TCP_PORT 12345

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

static volatile uint32_t global_seq_no = 0;

err_t send_response(struct tcp_pcb *tpcb, uint16_t msg_type,
                    const void *payload, uint32_t len);
err_t handle_actor_query(struct tcp_pcb *tpcb, const float *state);
err_t handle_minibatch_query(struct tcp_pcb *tpcb, const uint8_t *payload,
                             uint32_t len);
err_t handle_grad_update(struct tcp_pcb *tpcb, const uint8_t *payload,
                         uint32_t len);

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
  return err;
}

err_t handle_actor_query(struct tcp_pcb *tpcb, const float *state) {
  pl_start_forward(state);
  while (pl_is_busy()) {
    xemacif_input(app_netif);
  }
  float action, logpi;
  pl_read_result(&action, &logpi);
  char response[8];
  memcpy(response, &action, 4);
  memcpy(response + 4, &logpi, 4);
  return send_response(tpcb, MSG_TYPE_ACTOR_RESPONSE, response, 8);
}

err_t handle_minibatch_query(struct tcp_pcb *tpcb, const uint8_t *payload,
                             uint32_t len) {
  if (len < 4)
    return ERR_ARG;
  uint32_t N = *(uint32_t *)payload;
  if (N == 0 || N > 256)
    return ERR_ARG;
  const float *states = (const float *)(payload + 4);
  uint32_t resp_size = 4 + N * 8;
  char *resp = mem_malloc(resp_size);
  if (!resp)
    return ERR_MEM;
  *(uint32_t *)resp = N;
  float *out = (float *)(resp + 4);
  for (uint32_t i = 0; i < N; i++) {
    pl_start_forward(&states[i * 3]);
    while (pl_is_busy())
      xemacif_input(app_netif);
    pl_read_result(&out[0], &out[1]);
    out += 2;
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
  const float *grads = (const float *)(payload + 4);
  for (uint32_t i = 0; i < N; i++) {
    pl_start_backward(grads[i * 2], grads[i * 2 + 1]);
    while (pl_is_busy())
      xemacif_input(app_netif);
  }
  return send_response(tpcb, MSG_TYPE_ACK, NULL, 0);
}

err_t tcp_data_received(void *arg, struct tcp_pcb *tpcb, struct pbuf *p,
                        err_t err) {
  if (err != ERR_OK || p == NULL) {
    if (p)
      pbuf_free(p);
    tcp_close(tpcb);
    return ERR_OK;
  }

  if (p->tot_len >= sizeof(msg_header_t)) {
    msg_header_t hdr;
    if (pbuf_copy_partial(p, &hdr, sizeof(hdr), 0) != sizeof(hdr)) {
      tcp_recved(tpcb, p->tot_len);
      pbuf_free(p);
      return ERR_OK;
    }

    if (hdr.magic == MSG_MAGIC && hdr.version == 1) {
      uint32_t total_msg_len = sizeof(hdr) + hdr.payload_len;
      if (p->tot_len >= total_msg_len) {
        uint8_t *payload = NULL;
        if (hdr.payload_len > 0) {
          payload = mem_malloc(hdr.payload_len);
          if (!payload) {
            tcp_recved(tpcb, p->tot_len);
            pbuf_free(p);
            return ERR_OK;
          }
          if (pbuf_copy_partial(p, payload, hdr.payload_len, sizeof(hdr)) !=
              hdr.payload_len) {
            mem_free(payload);
            tcp_recved(tpcb, p->tot_len);
            pbuf_free(p);
            return ERR_OK;
          }
        }

        switch (hdr.msg_type) {
        case MSG_TYPE_ACTOR_QUERY:
          if (hdr.payload_len == 12) { // Only state[3] (3 floats)
            handle_actor_query(tpcb, (float *)payload);
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
          break;
        }

        if (payload)
          mem_free(payload);
      }
    }
  }

  tcp_recved(tpcb, p->tot_len);
  pbuf_free(p);
  return ERR_OK;
}

err_t tcp_connection_accepted(void *arg, struct tcp_pcb *newpcb, err_t err) {
  if (err != ERR_OK || newpcb == NULL)
    return ERR_VAL;
  tcp_arg(newpcb, NULL);
  tcp_recv(newpcb, tcp_data_received);
  return ERR_OK;
}

int init_tcp_server() {
  struct tcp_pcb *pcb = tcp_new();
  if (!pcb)
    return -1;
  err_t err = tcp_bind(pcb, IP_ADDR_ANY, TCP_PORT);
  if (err != ERR_OK) {
    tcp_close(pcb);
    return -1;
  }
  pcb = tcp_listen(pcb);
  if (!pcb)
    return -1;
  tcp_accept(pcb, tcp_connection_accepted);
  return 0;
}

int main() {
  // IP setup, lwIP init, netif add (same as original)
  ip_addr_t ipaddr, netmask, gw;
  IP4_ADDR(&ipaddr, 192, 168, 1, 10);
  IP4_ADDR(&netmask, 255, 255, 255, 0);
  IP4_ADDR(&gw, 192, 168, 1, 1);
  app_netif = &server_netif;

  init_platform();
  lwip_init();

  if (!xemac_add(app_netif, &ipaddr, &netmask, &gw,
                 (unsigned char[]){0x00, 0x0a, 0x35, 0x00, 0x01, 0x02},
                 PLATFORM_EMAC_BASEADDR)) {
    return -1;
  }
  netif_set_default(app_netif);
  netif_set_up(app_netif);

  if (init_tcp_server() != 0)
    return -1;
  xil_printf("SNN Actor (PL) ready on port %d\r\n", TCP_PORT);

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
  return 0;
}
