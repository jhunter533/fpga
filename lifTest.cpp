#include "LIF.h"
#include <iostream>

int main (int argc, char *argv[]) {
  fixed_t inputs[CONNECTIONS]={0.1,0.2,0.3,0.4,0.5};
  fixed_t weights[CONNECTIONS]={.5,.4,.3,.2,.1};
  fixed_t membrane=0;
  bool spike;
  for(int i=0;i<20;i++){
    lifNeuron(inputs,weights,membrane,spike);
    std::cout<<"Timestep "<<i<<": "<<"Membrane= "<<(float)membrane<<", Spike= "<<spike<<std::endl;
    if(i%5==0)inputs[0]+=(fixed_t).1;
  }
  return 0;
}
