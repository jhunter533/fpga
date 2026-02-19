#!/bin/bash

INTERFACE="enp6s0"

echo "Restoring DHCP"
sudo ip addr flush dev $INTERFACE

