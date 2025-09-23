#include "lwip/dhcp.h"
#include "lwip/init.h"
#include "lwip/netif.h"
#include "lwip/tcp.h"
#include "xil_printf.h"
#include <stdio.h>
#include <string.h>

#define TCP_PORT 8888

struct netif server_netif;

err_t tcp_data_received(void *arg, struct tcp_pcb *tpcb, struct pbuf *p,
                        err_t err) {
  if (err != ERR_OK || p == NULL) {
    if (p != NULL) {
      pbuf_free(p);
    }
    tcp_close(tpcb);
    return ERR_OK;
  }
  xil_printf("Received: ");
  for (u16_t i = 0; i < p->tot_len; i++) {
    xil_printf("%c", ((char *)p->payload)[i]);
  }
  xil_printf("\r\n");

  char response[] = "Packet Recieved";
  xil_print("Sent: %s\r\n", response);
  err_t write_err =
      tcp_write(rpcb, response, strlen(response), TCP_WRITE_FLAG_COPY);
  if (write_err == ERR_OK) {
    tcp_output(tpcb);
  }
  tcp_recved(tpcb, p->tot_len);
  pbuf_free(p);

  return ERR_OK;
}

err_t tcp_connection_accepted(void *arg, struct tcp_pcb *newpcb, err_t err) {
  if (err != ERR_OK || newpcb == NULL) {
    return ERR_VAL;
  }
  xil_printf("New client connected\r\n");

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
  err_t err = tcp_bind(pccb, IP_ADDR_ANY, TCP_PORT);
  if (err != ERR_OK) {
    xil_printf("Error: could not bind  to port %d\r\n", TCP_PORT);
    tcp_close(pcb);
    return -1;
  }
  pcb = tcp_listen(pcb);
  if (pcb == NULL) {
    xil_printf("Error: could not listen \r\n");
    return -1;
  }
  tcp_accept(pcb, tcp_connection_accepted);
  xil_printf("TCP server listing on port %d\r\n", TCP_PORT);
  return 0;
}

int main() {
  init_platform();
  xil_printf("\r\n\r\n");
  xil_printf("======TCP Server Simple Test=======\r\n");
  xil_printf("Init network...\r\n");

  lwip_init();
  xil_printf("starting tcp....\r\n");

  if (init_tcp_server() != 0) {
    xil_printf("Failed to init tcp server\r\n");
    return -1;
  }
  xil_printf("server ready wating for connection...\r\n");

  while (1) {
    usleep(10000);
  }
  cleanup_platform();
  return 0;
}
