#include "pl_driver.h"
#include "xil_io.h"
#include <stdint.h>

void pl_start_forward(const float state[3]) {
  Xil_Out32(PL_STATE_IN_0, *(uint32_t *)&state[0]);
  Xil_Out32(PL_STATE_IN_1, *(uint32_t *)&state[1]);
  Xil_Out32(PL_STATE_IN_2, *(uint32_t *)&state[2]);
  Xil_Out32(PL_CTRL_REG, CTRL_START_FWD);
}

void pl_start_backward(float grad_da, float grad_dlogp) {
  Xil_Out32(PL_GRAD_DA, *(uint32_t *)&grad_da);
  Xil_Out32(PL_GRAD_DLOGP, *(uint32_t *)&grad_dlogp);
  Xil_Out32(PL_CTRL_REG, CTRL_START_BWD);
}

int pl_is_busy(void) { return (Xil_In32(PL_STATUS_REG) & STATUS_BUSY) != 0; }

void pl_read_result(float *action, float *logpi) {
  *action = *(float *)Xil_In32(PL_ACTION_OUT);
  *logpi = *(float *)Xil_In32(PL_LOGPI_OUT);
}
