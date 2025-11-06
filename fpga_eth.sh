#!/bin/bash

INTERFACE="enp6s0"

echo "Setting static IP for FPGA"
sudo ip addr flush dev $INTERFACE
sudo ip addr add 192.168.1.100/24 dev $INTERFACE
sudo ip link set $INTERFACE up


