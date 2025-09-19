import struct
import time

import serial


class PacketTester:
    def __init__(self,port):
        self.ser=serial.Serial(port,baudrate=115200,timeout=5.0)
        print(f"connected to {port}")
    def send_packet(self,cmd,data=b''):
        packet=bytearray()
        packet.append(0xAA)
        packet.append(cmd)
        packet.append(len(data))
        packet.extend(data)
        checksum=sum(packet)&0xFF
        packet.append(checksum)
        print(f"Sending: {packet.hex()}")
        self.ser.write(packet)
    def receive_packet(self):
        start_time = time.time()
        timeout = 5.0  # seconds

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print("Timeout waiting for start byte")
                return None
            start_byte = self.ser.read(1)
            if start_byte == b'\xAA':
                break
            elif start_byte:
                print(f"Discarded byte: {start_byte.hex()}")
    
        cmd_byte = self.ser.read(1)
        if not cmd_byte:
            print("Missing command")
            return None
        cmd = cmd_byte[0]

        len_byte = self.ser.read(1)
        if not len_byte:
            print("Missing length")
            return None
        length = len_byte[0]

        data = b''
        if length > 0:
            data = self.ser.read(length)
            if len(data) != length:
                print(f"Data incomplete: expected {length}, got {len(data)}")
                return None

        checksum_byte = self.ser.read(1)
        if not checksum_byte:
            print("Missing checksum")
            return None
        recv_checksum = checksum_byte[0]

        calc_checksum = (0xAA + cmd + length + sum(data)) & 0xFF
        if calc_checksum != recv_checksum:
            print("Checksum error")
            return None

        print(f"Received: AA{cmd:02X}{length:02X}{data.hex()}{recv_checksum:02X}")
        return cmd, data
def main():
    tester=PacketTester('/tmp/virtual2')
    print("\nTest1")
    tester.send_packet(0x52)
    resp=tester.receive_packet()
    if resp:
        cmd,data=resp
        if cmd==0x44:
            print("Reset Successful")
    print("\n=== Test 2: Send reset command (0x52) ===")
    tester.send_packet(0x52)
    response = tester.receive_packet()
    if response:
        cmd, data = response
        print(f"Reset response: cmd={cmd:02X}")
    
    # Test 3: Data packet
    print("\n=== Test 3: Send data packet (0x46) ===")
    test_data = bytes([0x01, 0x02, 0x03, 0x04])
    tester.send_packet(0x46, test_data)
    response = tester.receive_packet()
    if response:
        cmd, data = response
        if data == test_data:
            print("Data echo successful!")

if __name__=="__main__":
    main()
