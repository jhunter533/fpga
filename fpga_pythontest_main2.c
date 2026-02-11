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
static struct netif server_netif;
struct netif *echo_netif;

void tcp_fasttmr(void);
void tcp_slowtmr(void);
#if LWIP_IPV6 == 1
#include "lwip/ip.h"
#else
#include "lwip/dhcp.h"
#endif

/* Echo server port */
#define TCP_PORT 8888

/* MAC address (must be unique) */
static unsigned char mac_ethernet_address[] = {0x00, 0x0a, 0x35,
                                               0x00, 0x01, 0x02};

/* Network interface */

struct netif *app_netif;

/* Forward declarations */
void print_app_header();
err_t tcp_data_received(void *arg, struct tcp_pcb *tpcb, struct pbuf *p,
                        err_t err);
err_t tcp_connection_accepted(void *arg, struct tcp_pcb *newpcb, err_t err);
int init_tcp_server();

/*------------------------------------------------------------------------*/
/* TCP Data Received Callback                                             */
/*------------------------------------------------------------------------*/
err_t tcp_data_received(void *arg, struct tcp_pcb *tpcb, struct pbuf *p,
                        err_t err) {
  if (err != ERR_OK || p == NULL) {
    if (p != NULL) {
      pbuf_free(p);
    }
    tcp_close(tpcb);
    return ERR_OK;
  }

  /* Print received data */
  xil_printf("Received: ");
  for (u16_t i = 0; i < p->tot_len; i++) {
    xil_printf("%c", ((char *)p->payload)[i]);
  }
  xil_printf("\r\n");

  /* Prepare response */
  const char *response = "Packet Received\r\n";
  xil_printf("Sent: %s", response);

  /* Send response */
  err_t write_err =
      tcp_write(tpcb, response, strlen(response), TCP_WRITE_FLAG_COPY);
  if (write_err == ERR_OK) {
    tcp_output(tpcb);
  }

  /* Update receive window */
  tcp_recved(tpcb, p->tot_len);

  /* Free buffer */
  pbuf_free(p);

  return ERR_OK;
}

/*------------------------------------------------------------------------*/
/* TCP Connection Accepted Callback                                       */
/*------------------------------------------------------------------------*/
err_t tcp_connection_accepted(void *arg, struct tcp_pcb *newpcb, err_t err) {
  if (err != ERR_OK || newpcb == NULL) {
    return ERR_VAL;
  }

  xil_printf("New client connected\r\n");

  tcp_arg(newpcb, NULL);
  tcp_recv(newpcb, tcp_data_received);

  return ERR_OK;
}

/*------------------------------------------------------------------------*/
/* Initialize TCP Server                                                  */
/*------------------------------------------------------------------------*/
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
  xil_printf("TCP server listening on port %d\r\n", TCP_PORT);
  return 0;
}

/*------------------------------------------------------------------------*/
/* Application Header Print                                               */
/*------------------------------------------------------------------------*/
void print_app_header() {
  xil_printf("\r\n\n\r\n");
  xil_printf("----- TCP Echo Server (Port %d) -----\r\n", TCP_PORT);
  xil_printf("Connect via: telnet 192.168.1.10 %d\r\n", TCP_PORT);
  xil_printf("-------------------------------------\r\n");
}

/*------------------------------------------------------------------------*/
/* Main Function                                                          */
/*------------------------------------------------------------------------*/
int main() {
  /* IP configuration */
#if LWIP_IPV6 == 0
  ip_addr_t ipaddr, netmask, gw;

  /* Static IP configuration (disable DHCP) */
  IP4_ADDR(&ipaddr, 192, 168, 1, 10);
  IP4_ADDR(&netmask, 255, 255, 255, 0);
  IP4_ADDR(&gw, 192, 168, 1, 1);
#endif

  app_netif = &server_netif;

  init_platform();

  print_app_header();

  /* Initialize lwIP */
  lwip_init();

#if LWIP_IPV6 == 0
  /* Add network interface */
  if (!xemac_add(app_netif, &ipaddr, &netmask, &gw, mac_ethernet_address,
                 PLATFORM_EMAC_BASEADDR)) {
    xil_printf("Error adding network interface\r\n");
    return -1;
  }
#else
  if (!xemac_add(app_netif, NULL, NULL, NULL, mac_ethernet_address,
                 PLATFORM_EMAC_BASEADDR)) {
    xil_printf("Error adding network interface\r\n");
    return -1;
  }
  app_netif->ip6_autoconfig_enabled = 1;
  netif_create_ip6_linklocal_address(app_netif, 1);
  netif_ip6_addr_set_state(app_netif, 0, IP6_ADDR_VALID);
#endif

  netif_set_default(app_netif);
  netif_set_up(app_netif);

  xil_printf("Network interface is UP.\r\n");

  /* Start TCP server */
  if (init_tcp_server() != 0) {
    xil_printf("Failed to initialize TCP server\r\n");
    return -1;
  }

  xil_printf("Server ready. Waiting for connections...\r\n");

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

    /* Small delay to avoid busy loop (optional) */
    usleep(1000); // 1ms
  }

  cleanup_platform();
  return 0;
}
