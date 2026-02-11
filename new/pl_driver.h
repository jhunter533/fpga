// pl_driver.h
#ifndef PL_DRIVER_H
#define PL_DRIVER_H

#include <stdint.h>

// REPLACE WITH YOUR ACTUAL BASE ADDRESS FROM VIVADO
#define PL_BASE_ADDR XPAR_SNN_ACCEL_0_S_AXI_CTRL_BASEADDR

// Register offsets (32-bit aligned)
#define PL_CTRL_REG (PL_BASE_ADDR + 0x00)
#define PL_STATUS_REG (PL_BASE_ADDR + 0x04)
#define PL_STATE_IN_0 (PL_BASE_ADDR + 0x08)
#define PL_STATE_IN_1 (PL_BASE_ADDR + 0x0C)
#define PL_STATE_IN_2 (PL_BASE_ADDR + 0x10)
#define PL_GRAD_DA (PL_BASE_ADDR + 0x14)
#define PL_GRAD_DLOGP (PL_BASE_ADDR + 0x18)
#define PL_ACTION_OUT (PL_BASE_ADDR + 0x1C)
#define PL_LOGPI_OUT (PL_BASE_ADDR + 0x20)

// Control register bits
#define CTRL_START_FWD (1U << 0)
#define CTRL_START_BWD (1U << 1)

// Status register bits
#define STATUS_BUSY (1U << 0)

// Function prototypes
void pl_start_forward(const float state[3]);
void pl_start_backward(float grad_da, float grad_dlogp);
int pl_is_busy(void);
void pl_read_result(float *action, float *logpi);

#endif
