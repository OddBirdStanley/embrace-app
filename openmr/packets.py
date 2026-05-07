import struct
import numpy as np

# Scaling constants (extracted from libMindRoveWrapper.so .rodata)
EXG_SCALE = 0.045            # uV per LSB, 24-bit ADC
ACCEL_SCALE = 6.1035e-05     # g per LSB (1/16384, +/-2g)
GYRO_SCALE = 0.01526         # deg/s per LSB (+/-500dps)
MAGNETOMETER_SCALE = 1.5     # uT per LSB

# Battery formula constants: (raw_mV / 1000 - 2.8) * 100 / 1.45
BATT_DIV1 = 1000.0
BATT_SUB = 2.8
BATT_MUL = 100.0
BATT_DIV2 = 1.45

# Packet type IDs (COMM_TYPE enum, first 2 bytes of non-WUA2 packets)
COMM_WUA2 = 1
COMM_MES2 = 2
COMM_IMP3 = 3
COMM_MES4 = 4
COMM_MES5 = 5
COMM_MES6 = 6
COMM_MES7 = 7

PACKET_SIZES = {
    COMM_WUA2: 216,
    COMM_MES2: 73,
    COMM_IMP3: 89,
    COMM_MES4: 75,
    COMM_MES5: 79,
    COMM_MES6: 77,
    COMM_MES7: 79,
}

# How many EEG sub-samples per packet
SUB_SAMPLES = {
    COMM_WUA2: 1,
    COMM_MES2: 2,
    COMM_IMP3: 2,
    COMM_MES4: 2,
    COMM_MES5: 2,
    COMM_MES6: 2,
    COMM_MES7: 2,
}


def _int24_be(buf, offset):
    """Read a 24-bit big-endian signed integer from 3 bytes."""
    b0, b1, b2 = buf[offset], buf[offset + 1], buf[offset + 2]
    val = (b0 << 16) | (b1 << 8) | b2
    if val & 0x800000:
        val -= 0x1000000
    return val


def _int16_le(buf, offset):
    """Read a 16-bit little-endian signed integer."""
    return struct.unpack_from('<h', buf, offset)[0]


def _uint16_le(buf, offset):
    """Read a 16-bit little-endian unsigned integer."""
    return struct.unpack_from('<H', buf, offset)[0]


def _int32_le(buf, offset):
    """Read a 32-bit little-endian signed integer."""
    return struct.unpack_from('<i', buf, offset)[0]


def _uint32_le(buf, offset):
    """Read a 32-bit little-endian unsigned integer."""
    return struct.unpack_from('<I', buf, offset)[0]


class PacketInterpreter:
    """Interprets raw MindRove device packets into channel data arrays.

    Usage:
        interp = PacketInterpreter(board_descr)
        for raw_bytes in socket_stream:
            samples = interp.parse(raw_bytes)
            # samples is a list of np.ndarray, one per sub-sample
    """

    def __init__(self, board_descr):
        self.descr = board_descr
        self.num_rows = board_descr["num_rows"]
        self.pkg_ch = board_descr.get("package_num_channel", 0)
        self.comm_type = None
        self.last_packet_num = -1
        self._lost_buffer = None

    def parse(self, raw):
        """Parse a raw UDP/TCP payload into a list of sample arrays.

        Returns list of np.ndarray, each of shape (num_rows,).
        For WUA2: 1 sample. For MES/IMP types: 2 samples.
        """
        data = bytes(raw)

        # WUA2 auto-detect by size (no 2-byte prefix)
        if len(data) == PACKET_SIZES[COMM_WUA2]:
            self.comm_type = COMM_WUA2
            return self._parse_wua2(data)

        # Config packet (5 bytes)
        if len(data) == 5:
            return []

        # Non-WUA2: first 2 bytes are COMM_TYPE
        if len(data) < 2:
            return []

        comm_type = struct.unpack_from('<h', data, 0)[0]
        if comm_type not in PACKET_SIZES:
            return []

        self.comm_type = comm_type
        payload = data[2:]
        expected = PACKET_SIZES[comm_type]

        if len(payload) != expected:
            return []

        n_sub = SUB_SAMPLES[comm_type]
        samples = []
        for idx in range(n_sub):
            out = np.zeros(self.num_rows, dtype=np.float64)
            if comm_type == COMM_MES2:
                self._interpret_mes2(payload, out, idx)
            elif comm_type == COMM_IMP3:
                self._interpret_imp3(payload, out, idx)
            elif comm_type == COMM_MES4:
                self._interpret_mes4(payload, out, idx)
            elif comm_type == COMM_MES5:
                self._interpret_mes5(payload, out, idx)
            elif comm_type == COMM_MES6:
                self._interpret_mes6(payload, out, idx)
            elif comm_type == COMM_MES7:
                self._interpret_mes7(payload, out, idx)
            samples.append(out)
        return samples

    def _parse_wua2(self, data):
        """WUA2: 216 bytes = 54 x int32 LE, single sample."""
        out = np.zeros(self.num_rows, dtype=np.float64)
        ints = struct.unpack_from('<54i', data, 0)

        # exg_channels[0..7] at int offsets 0-7, scale=0.045
        for i, ch in enumerate(self.descr.get("exg_channels", [])[:8]):
            out[ch] = ints[i] * EXG_SCALE

        # resistance_channels[0..9] at int offsets 8-17, scale=1.0
        for i, ch in enumerate(self.descr.get("resistance_channels", [])[:10]):
            out[ch] = float(ints[8 + i])

        # battery_channel at int offset 18: (raw/1000 - 2.8) * 100 / 1.45
        batt_ch = self.descr.get("battery_channel")
        if batt_ch is not None:
            raw = ints[18]
            out[batt_ch] = (raw / BATT_DIV1 - BATT_SUB) * BATT_MUL / BATT_DIV2

        # other_channels[0] at int offset 19, scale=1.0
        other = self.descr.get("other_channels", [])
        if len(other) >= 1:
            out[other[0]] = float(ints[19])

        # accel_channels[0..2] at int offsets 20-22, scale=6.1035e-05
        for i, ch in enumerate(self.descr.get("accel_channels", [])[:3]):
            out[ch] = ints[20 + i] * ACCEL_SCALE

        # gyro_channels[0..2] at int offsets 23-25, scale=0.01526
        for i, ch in enumerate(self.descr.get("gyro_channels", [])[:3]):
            out[ch] = ints[23 + i] * GYRO_SCALE

        # package_num_channel at int offset 26
        out[self.pkg_ch] = float(ints[26])

        # other_channels[1] from object state (internal counter)
        if len(other) >= 2:
            out[other[1]] = 0.0  # no internal state in pure python

        return [out]

    def _read_exg_24bit(self, buf, out, base_offset):
        """Read 8 EXG channels as 24-bit BE signed, scale by 0.045."""
        for i, ch in enumerate(self.descr.get("exg_channels", [])[:8]):
            out[ch] = _int24_be(buf, base_offset + i * 3) * EXG_SCALE

    def _read_accel(self, buf, out, offset):
        """Read 3 accel channels as int16 LE."""
        for i, ch in enumerate(self.descr.get("accel_channels", [])[:3]):
            out[ch] = _int16_le(buf, offset + i * 2) * ACCEL_SCALE

    def _read_gyro(self, buf, out, offset):
        """Read 3 gyro channels as int16 LE."""
        for i, ch in enumerate(self.descr.get("gyro_channels", [])[:3]):
            out[ch] = _int16_le(buf, offset + i * 2) * GYRO_SCALE

    def _read_battery_byte(self, buf, out, offset):
        """Read battery as single unsigned byte."""
        batt_ch = self.descr.get("battery_channel")
        if batt_ch is not None:
            out[batt_ch] = float(buf[offset])

    def _read_other0_byte(self, buf, out, offset):
        """Read other_channels[0] as unsigned byte."""
        other = self.descr.get("other_channels", [])
        if len(other) >= 1:
            out[other[0]] = float(buf[offset])

    def _read_nibbles(self, buf, out, offset):
        """Read other_channels[2] and [3] from nibble-split byte."""
        other = self.descr.get("other_channels", [])
        if len(other) >= 4:
            b = buf[offset]
            out[other[2]] = float(b >> 4)
            out[other[3]] = float(b & 0x0F)

    def _read_pkg_num_u32(self, buf, out, offset):
        """Read package_num as uint32 LE."""
        out[self.pkg_ch] = float(_uint32_le(buf, offset))

    # --- MES_2: 73 bytes, 2 sub-samples ---
    # EXG: 8ch x 3B x 2 = 48B at 0x00, stride 24
    # battery: 1B at 0x30
    # other[0]: 1B at 0x31+idx
    # nibbles: 1B at 0x33+idx
    # accel: 3x int16 at 0x35
    # gyro: 3x int16 at 0x3B
    # pkg_num: uint32 at 0x41+idx*4
    def _interpret_mes2(self, buf, out, idx):
        self._read_exg_24bit(buf, out, idx * 24)
        self._read_battery_byte(buf, out, 0x30)
        self._read_other0_byte(buf, out, 0x31 + idx)
        self._read_nibbles(buf, out, 0x33 + idx)
        self._read_accel(buf, out, 0x35)
        self._read_gyro(buf, out, 0x3B)
        self._read_pkg_num_u32(buf, out, 0x41 + idx * 4)

    # --- IMP_3: 89 bytes, 2 sub-samples ---
    # resistance: 10x uint32 x 2 = 80B at 0x00, stride 40
    # battery: 1B at 0x50
    # pkg_num: uint32 at 0x51+idx*4
    def _interpret_imp3(self, buf, out, idx):
        base = idx * 40
        for i, ch in enumerate(self.descr.get("resistance_channels", [])[:10]):
            val = _int32_le(buf, base + i * 4)
            out[ch] = float(val) if val >= 0 else 0.0
        self._read_battery_byte(buf, out, 0x50)
        self._read_pkg_num_u32(buf, out, 0x51 + idx * 4)

    # --- MES_4: 75 bytes, 2 sub-samples ---
    # EXG: 8ch x 3B x 2 = 48B at 0x00, stride 24
    # battery: 1B at 0x30
    # other[0]: 1B at 0x31+idx
    # nibbles: 1B at 0x33+idx
    # accel: 3x int16 at 0x35
    # gyro: 3x int16 at 0x3B
    # ppg: 2x uint8 at 0x41
    # pkg_num: uint32 at 0x43+idx*4
    def _interpret_mes4(self, buf, out, idx):
        self._read_exg_24bit(buf, out, idx * 24)
        self._read_battery_byte(buf, out, 0x30)
        self._read_other0_byte(buf, out, 0x31 + idx)
        self._read_nibbles(buf, out, 0x33 + idx)
        self._read_accel(buf, out, 0x35)
        self._read_gyro(buf, out, 0x3B)
        for i, ch in enumerate(self.descr.get("ppg_channels", [])[:2]):
            out[ch] = float(buf[0x41 + i])
        self._read_pkg_num_u32(buf, out, 0x43 + idx * 4)

    # --- MES_5: 79 bytes, 2 sub-samples ---
    # Same as MES_4 but ppg_raw_channels: 3x int16 at 0x41
    # pkg_num: uint32 at 0x47+idx*4
    def _interpret_mes5(self, buf, out, idx):
        self._read_exg_24bit(buf, out, idx * 24)
        self._read_battery_byte(buf, out, 0x30)
        self._read_other0_byte(buf, out, 0x31 + idx)
        self._read_nibbles(buf, out, 0x33 + idx)
        self._read_accel(buf, out, 0x35)
        self._read_gyro(buf, out, 0x3B)
        for i, ch in enumerate(self.descr.get("ppg_raw_channels", [])[:3]):
            out[ch] = float(_int16_le(buf, 0x41 + i * 2))
        self._read_pkg_num_u32(buf, out, 0x47 + idx * 4)

    # --- MES_6: 77 bytes, 2 sub-samples ---
    # Same EXG/battery/other/accel/gyro as MES_2
    # eda: uint32 at 0x49
    # pkg_num: uint32 at 0x41+idx*4
    def _interpret_mes6(self, buf, out, idx):
        self._read_exg_24bit(buf, out, idx * 24)
        self._read_battery_byte(buf, out, 0x30)
        self._read_other0_byte(buf, out, 0x31 + idx)
        self._read_nibbles(buf, out, 0x33 + idx)
        self._read_accel(buf, out, 0x35)
        self._read_gyro(buf, out, 0x3B)
        self._read_pkg_num_u32(buf, out, 0x41 + idx * 4)
        for ch in self.descr.get("eda_channels", [])[:1]:
            out[ch] = float(_uint32_le(buf, 0x49))

    # --- MES_7: 79 bytes, 2 sub-samples ---
    # Same EXG/battery/other/accel/gyro as MES_2
    # magnetometer: 3x int16 at 0x41, scale=1.5
    # pkg_num: uint32 at 0x47+idx*4
    def _interpret_mes7(self, buf, out, idx):
        self._read_exg_24bit(buf, out, idx * 24)
        self._read_battery_byte(buf, out, 0x30)
        self._read_other0_byte(buf, out, 0x31 + idx)
        self._read_nibbles(buf, out, 0x33 + idx)
        self._read_accel(buf, out, 0x35)
        self._read_gyro(buf, out, 0x3B)
        for i, ch in enumerate(self.descr.get("magnetometer_channels", [])[:3]):
            out[ch] = float(_int16_le(buf, 0x41 + i * 2)) * MAGNETOMETER_SCALE
        self._read_pkg_num_u32(buf, out, 0x47 + idx * 4)
