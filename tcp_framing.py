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
    TYPE_GRAD_UPDATE=5
    TYPE_ACK=6
    TYPE_PING = 7
    TYPE_PONG = 8
    
    def __init__(self,msg_type:int,seq:int,payload:bytes=b""):
        self.msg_type=msg_type
        self.seq=seq
        self.payload=payload
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "Message":
        if len(data) < cls.HEADER_SIZE:
            raise ValueError("Incomplete header")
        magic, version, msg_type, seq, payload_len = struct.unpack(
            cls.HEADER_FMT, data[:cls.HEADER_SIZE]
        )
        if magic != cls.MAGIC:
            raise ValueError(f"Bad magic: 0x{magic:08X} (expected 0x{cls.MAGIC:08X})")
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")
        expected_total = cls.HEADER_SIZE + payload_len
        if len(data) < expected_total:
            raise ValueError(f"Incomplete message: got {len(data)} bytes, need {expected_total}")
        payload = data[cls.HEADER_SIZE : expected_total]
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
    
    @classmethod
    def grad_update(cls, seq: int, dL_da_list: List[List[float]], dL_dlogp_list: List[float]) -> "Message":
        B = len(dL_da_list)
        assert B == len(dL_dlogp_list)
        if B == 0:
            raise ValueError("Empty gradient batch")
        A = len(dL_da_list[0])
        payload = bytearray()
        payload.extend(struct.pack("<I", B))
        for i in range(B):
            payload.extend(struct.pack("<" + "f" * A, *dL_da_list[i]))
            payload.extend(struct.pack("<f", dL_dlogp_list[i]))
        return cls(cls.TYPE_GRAD_UPDATE, seq, bytes(payload))

    @classmethod
    def ping(cls, seq: int) -> "Message":
        return cls(cls.TYPE_PING, seq)

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
    def expect_ack(self) -> None:
        if self.msg_type != self.TYPE_ACK:
            raise ValueError(f"Expected ACK (6), got {self.msg_type}")

    def expect_pong(self) -> None:
        if self.msg_type != self.TYPE_PONG:
            raise ValueError(f"Expected PONG (8), got {self.msg_type}")

