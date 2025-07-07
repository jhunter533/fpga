import os
import random
import struct
import time

import gymnasium as gym
import numpy as np
import serial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FPGAInterface:
    def __init__(self,port,baudrate=115200):
        self.ser=serial.Serial(port,baudrate,timeout=1)
        time.sleep(2)

    def float_to_fixed(self,x):
        scale=2**8
        xVal=float(x)
        if xVal<-128:
            xVal=-128
        elif xVal>127.996:
            xVal=127.996
        xInt=int(xVal*scale)&0xFFFF
        return xInt

    def fixed_to_float(self,x):
        scale=2**8
        if x>0x7FFF:
            x-=0x10000
        return x/scale
    
    def sendCommand(self,command: bytes):
        self.ser.write(command)

    def sendState(self,state: np.ndarray):
        self.ser.write(b'F')
        for value in state:
            fixed=self.float_to_fixed(value)
            self.ser.write(struct.pack('<H',fixed))

    def getAction(self)->float:
        resp=self.ser.read(2)
        if len(resp)!=2:
            return 0.0
        fixed=struct.unpack('<H',resp)[0]
        return self.fixed_to_float(fixed)

    def sendReward(self,reward):
        self.ser.write(b'B')
        fixed=self.float_to_fixed(reward)
        self.ser.write(struct.pack('<H',fixed))
        return self.waitForAck()

    def resetNeurons(self):
        self.ser.write(b'R')
        return self.waitForAck()

    def waitForAck(self,timeout=1.0):
        start=time.time()
        while time.time()-start<timeout:
            if self.ser.in_waiting>0:
                resp=self.ser.read(1)
                return resp == b'D'
        return False

def main():
    env=gym.make("Pendulum-v1",render_mode='human')
    fpga=FPGAInterface('/dev/ttyUSB1')#check usb port
    numEpisodes=1000
    gamma=.99

    for episode in range(numEpisodes):
        state,_=env.reset()
        fpga.resetNeurons()
        states,actions,rewards=[],[],[]
        totalReward=0
        done=False

        while not done:
            fpga.sendState(state)
            action=fpga.getAction()
            nextState,reward,terminated,truncated,_=env.step([action])
            done=terminated or truncated
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state=nextState
            totalReward+=reward
        discountReward=0
        for t in reversed(range(len(rewards))):
            discountReward=rewards[t]+gamma*discountReward
            rewards[t]=discountReward
        rewards=np.array(rewards)
        rewards=(rewards-rewards.mean())/(rewards.std()+1e-8)
        for i, (state,action,reward) in enumerate(zip(states,actions,rewards)):
            #fpga.sendReward(reward)
            print(f"Epsiode{episode+1}/{numEpisodes},totalReward: {totalReward}")
    env.close()
if __name__=="__main__":
    main()
            
