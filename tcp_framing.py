import socket
import struct
from typing import List, Optional, Tuple


class Message:
    MAGIC=0xDEADBEEF
    HEADER_FMT="<IHHII"
    HEADER_SIZE=struct.calcsize(HEADER_FMT)
    TYPE_ACTOR_QUERY=1
    TYPE_ACTOR_RESPONSE = 2
    TYPE_MINIBATCH_QUERY = 3
    TYPE_MINIBATCH_RESP = 4
    TYPE_PING = 6
    TYPE_PONG = 7
    
    def __init__(self,msg_type:int,seq:int,payload:bytes=b""):
        self.msg_type=msg_type
        self.seq=seq
        self.payload=payload
    
    @classmethod
    def from_bytes(cls,data:bytes)->"Message":
        if len(data)<cls.HEADER_SIZE:
            raise ValueError("Incomplete header")
        magic,version,msg_type,seq,payload_len=struct.unpack(cls.HEADER_FMT,data[:cls.HEADER_SIZE])
        if magic!=cls.MAGIC:
            raise ValueError(f"Bad magic: {hex(magic)}")
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")
        if len(data) < cls.HEADER_SIZE + payload_len:
            raise ValueError("Incomplete payload")
        payload = data[cls.HEADER_SIZE : cls.HEADER_SIZE + payload_len]
        return cls(msg_type, seq, payload)
    
    def to_bytes(self) -> bytes:
        header = struct.pack(
            self.HEADER_FMT,
            self.MAGIC,
            1,  # version
            self.msg_type,
            self.seq,
            len(self.payload),
        )
        return header + self.payload
    
    @classmethod
    def actor_query(cls, seq: int, state: List[float], done: bool) -> "Message":
        payload = struct.pack("<" + "f" * (len(state) + 1), *state, 1.0 if done else 0.0)
        return cls(cls.TYPE_ACTOR_QUERY, seq, payload)
    
    @classmethod
    def minibatch_query(cls, seq: int, states: List[List[float]]) -> "Message":
        N = len(states)
        flat = [val for s in states for val in s]  # flatten
        payload = struct.pack("<I" + "f" * len(flat), N, *flat)
        return cls(cls.TYPE_MINIBATCH_QUERY, seq, payload)
    
    def parse_actor_response(self, num_actions=1) -> Tuple[List[float], float]:
        expected = 4 * (num_actions + 1)
        if len(self.payload) != expected:
            raise ValueError(f"Wrong actor resp size: {len(self.payload)} vs {expected}")
        data = struct.unpack("<" + "f" * (num_actions + 1), self.payload)
        return list(data[:-1]), data[-1]  # action list, log_prob

    
    def parse_minibatch_response(self, num_actions=1) -> List[Tuple[List[float], float]]:
        if len(self.payload) < 4:
            raise ValueError("Minibatch resp too short")
        N = struct.unpack("<I", self.payload[:4])[0]
        expected_size = 4 + N * 4 * (num_actions + 1)
        if len(self.payload) != expected_size:
            raise ValueError(f"Minibatch size mismatch: got {len(self.payload)}, expected {expected_size}")
        results = []
        offset = 4
        for _ in range(N):
            data = struct.unpack_from("<" + "f" * (num_actions + 1), self.payload, offset)
            action = list(data[:-1])
            log_prob = data[-1]
            results.append((action, log_prob))
            offset += 4 * (num_actions + 1)
        return results

