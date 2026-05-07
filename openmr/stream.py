import socket
import struct
import threading
import time
import numpy as np
from collections import deque

from .board_metadata import get_board_descr, get_sampling_rate
from .packets import PacketInterpreter


class RingBuffer:
    """Thread-safe ring buffer for EEG samples."""

    def __init__(self, num_rows, capacity=450000):
        self._lock = threading.Lock()
        self._buf = np.zeros((num_rows, capacity), dtype=np.float64)
        self._num_rows = num_rows
        self._capacity = capacity
        self._write_pos = 0
        self._count = 0

    def push(self, sample):
        """Push a single sample (num_rows,) into the buffer."""
        with self._lock:
            pos = self._write_pos % self._capacity
            self._buf[:, pos] = sample
            self._write_pos += 1
            if self._count < self._capacity:
                self._count += 1

    @property
    def count(self):
        with self._lock:
            return self._count

    def get_all(self):
        """Get and drain all available samples. Returns (num_rows, n_samples)."""
        with self._lock:
            n = self._count
            if n == 0:
                return np.zeros((self._num_rows, 0), dtype=np.float64)
            start = (self._write_pos - n) % self._capacity
            if start + n <= self._capacity:
                data = self._buf[:, start:start + n].copy()
            else:
                tail = self._capacity - start
                data = np.hstack([
                    self._buf[:, start:self._capacity],
                    self._buf[:, :n - tail],
                ])
            self._count = 0
            return data

    def get_current(self, n_samples):
        """Peek at the most recent n_samples without draining."""
        with self._lock:
            n = min(n_samples, self._count)
            if n == 0:
                return np.zeros((self._num_rows, 0), dtype=np.float64)
            start = (self._write_pos - n) % self._capacity
            if start + n <= self._capacity:
                return self._buf[:, start:start + n].copy()
            tail = self._capacity - start
            return np.hstack([
                self._buf[:, start:self._capacity],
                self._buf[:, :n - tail],
            ])


class MindRoveStream:
    """Pure Python streaming client for MindRove EEG devices.

    WiFi board:
        stream = MindRoveStream(board_id=0, ip="192.168.4.1")
        stream.start()
        data = stream.get_data()   # (num_rows, n_samples)
        stream.stop()

    SyncBox:
        stream = MindRoveStream(board_id=1, ip="syncbox.local")
        stream.start()
        data = stream.get_data()
        stream.stop()

    As context manager:
        with MindRoveStream(board_id=0) as s:
            s.start()
            while True:
                data = s.get_data()
                if data.shape[1] > 0:
                    process(data)
                time.sleep(0.1)

    As iterator:
        for chunk in MindRoveStream(board_id=0, chunk_size=250):
            process(chunk)   # (num_rows, 250)
    """

    def __init__(self, board_id=0, ip=None, port=None, preset=0,
                 buffer_size=450000, chunk_size=None,
                 stream_timeout=None, on_disconnect=None):
        """
        Args:
            stream_timeout: seconds of silence before treating the stream as lost.
                            None disables mid-stream timeout detection.
            on_disconnect: callback(reason_str) invoked on stream timeout.
                           If None, a ConnectionError is stored and raised on next get_data().
        """
        self.board_id = board_id
        self.preset = preset
        self.descr = get_board_descr(board_id, preset)
        self.num_rows = self.descr["num_rows"]
        self.sampling_rate = self.descr.get("sampling_rate", 500)
        self.chunk_size = chunk_size
        self.stream_timeout = stream_timeout
        self.on_disconnect = on_disconnect

        # Connection params
        if board_id == 0:  # WiFi
            self.ip = ip or "192.168.4.1"
            self.port = port or 4210
            self._transport = "udp"
        elif board_id == 1:  # SyncBox
            self.ip = ip or "syncbox.local"
            self.port = port or 5005
            self._transport = "tcp"
        else:
            raise ValueError(f"board_id {board_id} does not support streaming")

        self._interpreter = PacketInterpreter(self.descr)
        self._ring = RingBuffer(self.num_rows, buffer_size)
        self._recv_thread = None
        self._running = False
        self._sock = None
        self._recv_sock = None
        self._stats = {"packets": 0, "samples": 0, "errors": 0}
        self._last_packet_time = None
        self._disconnected_error = None

    def start(self, timeout=5.0):
        """Connect to device and start receiving data.

        Args:
            timeout: seconds to wait for first packet (UDP only). Set to 0 to skip.

        Raises:
            ConnectionError: if no data received within timeout (device not found).
        """
        if self._running:
            return

        if self._transport == "udp":
            self._start_udp()
        elif self._transport == "tcp":
            self._start_tcp()

        self._running = True
        self._recv_thread = threading.Thread(
            target=self._recv_loop, daemon=True, name="mindrove-recv"
        )
        self._recv_thread.start()

        if self._transport == "udp" and timeout > 0:
            deadline = time.time() + timeout
            while time.time() < deadline:
                if self._stats["packets"] > 0:
                    return
                time.sleep(0.1)
            self.stop()
            raise ConnectionError(
                f"No data received from {self.ip}:{self.port} within {timeout}s — "
                f"device not found on network"
            )

    def stop(self):
        """Stop receiving and close sockets."""
        self._running = False
        if self._recv_thread:
            self._recv_thread.join(timeout=3.0)
            self._recv_thread = None
        self._close_sockets()

    def get_data(self, n_samples=None):
        """Get available data from the ring buffer.

        Returns np.ndarray of shape (num_rows, n_samples).
        If n_samples is None, returns all available data (drains buffer).

        Raises:
            ConnectionError: if stream_timeout elapsed with no packets.
        """
        if self._disconnected_error:
            err = self._disconnected_error
            self._disconnected_error = None
            raise err
        if n_samples is None:
            return self._ring.get_all()
        return self._ring.get_current(n_samples)

    @property
    def data_count(self):
        """Number of samples currently in the buffer."""
        return self._ring.count

    @property
    def stats(self):
        return dict(self._stats)

    # --- Transport setup ---

    def _start_udp(self):
        """WiFi direct: UDP recv on port 4210."""
        self._recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._recv_sock.bind(("0.0.0.0", self.port))
        self._recv_sock.settimeout(2.0)

        # Send socket (for config commands)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(2.0)

    def _start_tcp(self):
        """SyncBox: TCP6 to syncbox.local:5005, Hello handshake."""
        infos = socket.getaddrinfo(
            self.ip, self.port, socket.AF_INET6, socket.SOCK_STREAM
        )
        connected = False
        for af, socktype, proto, canonname, sa in infos:
            try:
                self._sock = socket.socket(af, socktype, proto)
                self._sock.settimeout(5.0)
                self._sock.connect(sa)
                connected = True
                break
            except OSError:
                if self._sock:
                    self._sock.close()
                    self._sock = None

        if not connected:
            # Fall back to IPv4
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(5.0)
            self._sock.connect((self.ip, self.port))

        # Handshake
        self._sock.send(b"Hello")
        resp = self._sock.recv(128)
        if not resp.startswith(b"OK"):
            self._sock.close()
            raise ConnectionError(
                f"SyncBox handshake failed: {resp!r}"
            )
        self._recv_sock = self._sock

    def _close_sockets(self):
        for s in (self._recv_sock, self._sock):
            if s:
                try:
                    s.close()
                except OSError:
                    pass
        self._recv_sock = None
        self._sock = None

    # --- Receive loop ---

    def _recv_loop(self):
        if self._transport == "udp":
            self._recv_loop_udp()
        else:
            self._recv_loop_tcp()

    def _check_stream_timeout(self):
        """Check if stream has gone silent beyond stream_timeout."""
        if self.stream_timeout is None or self._last_packet_time is None:
            return
        elapsed = time.time() - self._last_packet_time
        if elapsed >= self.stream_timeout:
            reason = (
                f"No data received for {elapsed:.1f}s "
                f"(timeout={self.stream_timeout}s) — device may have disconnected"
            )
            if self.on_disconnect:
                self.on_disconnect(reason)
            else:
                self._disconnected_error = ConnectionError(reason)
            self._running = False

    def _recv_loop_udp(self):
        """WiFi: receive UDP datagrams, each is one packet."""
        while self._running:
            try:
                data, addr = self._recv_sock.recvfrom(512)
            except socket.timeout:
                self._check_stream_timeout()
                continue
            except OSError:
                break

            self._last_packet_time = time.time()
            self._stats["packets"] += 1
            try:
                samples = self._interpreter.parse(data)
                for s in samples:
                    self._ring.push(s)
                    self._stats["samples"] += 1
            except Exception:
                self._stats["errors"] += 1

    def _recv_loop_tcp(self):
        """SyncBox: receive TCP stream, parse 10-byte framed packets."""
        buf = bytearray()
        while self._running:
            try:
                chunk = self._recv_sock.recv(65536)
            except socket.timeout:
                self._check_stream_timeout()
                continue
            except OSError:
                break

            if not chunk:
                break

            self._last_packet_time = time.time()
            buf.extend(chunk)

            # Parse syncbox framing: [ssid:4][unk:2][size:4][payload:size]
            while len(buf) >= 10:
                if len(buf) < 10:
                    break
                pkt_size = struct.unpack_from('<i', buf, 6)[0]
                if pkt_size < 0:
                    # Corrupted, skip 1 byte
                    buf.pop(0)
                    self._stats["errors"] += 1
                    continue
                total = 10 + pkt_size
                if len(buf) < total:
                    break

                payload = bytes(buf[10:total])
                buf = buf[total:]

                self._stats["packets"] += 1
                try:
                    samples = self._interpreter.parse(payload)
                    for s in samples:
                        self._ring.push(s)
                        self._stats["samples"] += 1
                except Exception:
                    self._stats["errors"] += 1

    # --- Config ---

    def send_config(self, command):
        """Send a config command to the device."""
        if not self._sock:
            raise RuntimeError("Not connected")
        cmd = command.encode() if isinstance(command, str) else command
        if self._transport == "udp":
            self._sock.sendto(cmd, (self.ip, self.port))
        else:
            self._sock.send(cmd)

    # --- Context manager ---

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()

    # --- Iterator (yields chunks) ---

    def __iter__(self):
        if not self._running:
            self.start()
        chunk = self.chunk_size or self.sampling_rate
        try:
            while self._running:
                if self._ring.count >= chunk:
                    yield self._ring.get_all()
                else:
                    time.sleep(chunk / self.sampling_rate / 2)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
