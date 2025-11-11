import socket
import struct
import threading
import time
from typing import List, Tuple

from tcp_framing import Message


class FramedSocket:
    def __init__(self,host:str,port:int,timeout: float=2.0):
        self.host=host
        self.port=port
        self.timeout=timeout
        self.seq=0
        self.sock=None
        self.lock=threading.Lock()
        self.connect()
    def connect(self):
        with self.lock:
            if self.sock:
                self.sock.close()
            self.sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect((self.host,self.port))
            self.seq=0
    def send_msg(self,msg: Message)->None:
        with self.lock:
            if not self.sock:
                raise ConnectionError("Not connected")
            try:
                data=msg.to_bytes()
                self.sock.sendall(data)
            except (OSError, socket.timeout) as e:
                self.sock.close()
                self.sock=None
                raise ConnectionError(f"Send failed: {e}")
    def recv_msg(self)->Message:
        with self.lock:
            if not self.sock:
                raise ConnectionError("Not connected")
            try:
                # Read header
                header_data = b""
                while len(header_data) < Message.HEADER_SIZE:
                    chunk = self.sock.recv(Message.HEADER_SIZE - len(header_data))
                    if not chunk:
                        raise ConnectionError("Connection closed")
                    header_data += chunk
                
                magic, version, msg_type, seq, payload_len = struct.unpack(
                    Message.HEADER_FMT, header_data
                )
                if magic != Message.MAGIC:
                    raise ValueError(f"Bad magic in header: {hex(magic)}")

                # Read payload
                payload = b""
                while len(payload) < payload_len:
                    chunk = self.sock.recv(payload_len - len(payload))
                    if not chunk:
                        raise ConnectionError("Connection closed mid-payload")
                    payload += chunk

                return Message(msg_type, seq, payload)

            except (OSError, socket.timeout) as e:
                self.sock.close()
                self.sock = None
                raise ConnectionError(f"Receive failed: {e}")
    def query_actor(self, state, done) -> Tuple[List[float], float]:
        self.seq += 1
        msg = Message.actor_query(self.seq, state, done)
        self.send_msg(msg)
        resp = self.recv_msg()
        if resp.msg_type != Message.TYPE_ACTOR_RESPONSE:
            raise ValueError(f"Expected actor response, got {resp.msg_type}")
        return resp.parse_actor_response(num_actions=1)  # for Pendulum

    def query_minibatch(self, states: List[List[float]]) -> List[Tuple[List[float], float]]:
        self.seq += 1
        msg = Message.minibatch_query(self.seq, states)
        self.send_msg(msg)
        resp = self.recv_msg()
        if resp.msg_type != Message.TYPE_MINIBATCH_RESP:
            raise ValueError(f"Expected minibatch response, got {resp.msg_type}")
        return resp.parse_minibatch_response(num_actions=1)
