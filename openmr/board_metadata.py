BOARD_DESCRIPTORS = {
    -3: {
        0: {"name": "PlayBack"},
        1: {"name": "PlayBack"},
        2: {"name": "PlayBack"},
    },
    -2: {
        0: {"name": "Streaming"},
        1: {"name": "Streaming"},
        2: {"name": "Streaming"},
    },
    -1: {
        0: {
            "accel_channels": [17, 18, 19],
            "battery_channel": 29,
            "ecg_channels": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            "eda_channels": [23],
            "eeg_channels": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            "eeg_names": "Fz,C3,Cz,C4,Pz,PO7,Oz,PO8,F5,F7,F3,F1,F2,F4,F6,F8",
            "emg_channels": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            "eog_channels": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            "gyro_channels": [20, 21, 22],
            "marker_channel": 31,
            "name": "Synthetic",
            "num_rows": 34,
            "package_num_channel": 0,
            "ppg_channels": [24, 25],
            "ppg_raw_channels": [32, 33],
            "resistance_channels": [27, 28],
            "sampling_rate": 250,
            "temperature_channels": [26],
            "timestamp_channel": 30,
        },
        1: {
            "accel_channels": [2, 3, 4],
            "battery_channel": 1,
            "eda_channels": [8],
            "gyro_channels": [5, 6, 7],
            "marker_channel": 19,
            "name": "SyntheticAux",
            "num_rows": 20,
            "other_channels": [14, 15, 16, 17],
            "package_num_channel": 0,
            "ppg_channels": [9, 10],
            "resistance_channels": [12, 13],
            "sampling_rate": 250,
            "temperature_channels": [11],
            "timestamp_channel": 18,
        },
    },
    0: {
        0: {
            "accel_channels": [20, 21, 22],
            "battery_channel": 18,
            "eda_channels": [38],
            "eeg_channels": [0, 1, 2, 3, 4, 5, 6, 7],
            "emg_channels": [0, 1, 2, 3, 4, 5, 6, 7],
            "exg_channels": [0, 1, 2, 3, 4, 5, 6, 7],
            "gyro_channels": [23, 24, 25],
            "magnetometer_channels": [35, 36, 37],
            "marker_channel": 28,
            "name": "MindRoveWifi",
            "num_rows": 39,
            "other_channels": [19, 29],
            "package_num_channel": 26,
            "ppg_channels": [30, 31],
            "ppg_raw_channels": [32, 33, 34],
            "resistance_channels": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            "sampling_rate": 500,
            "timestamp_channel": 27,
        },
    },
    1: {
        0: {
            "accel_channels": [20, 21, 22],
            "battery_channel": 18,
            "eda_channels": [40],
            "eeg_channels": [0, 1, 2, 3, 4, 5, 6, 7],
            "emg_channels": [0, 1, 2, 3, 4, 5, 6, 7],
            "exg_channels": [0, 1, 2, 3, 4, 5, 6, 7],
            "gyro_channels": [23, 24, 25],
            "magnetometer_channels": [37, 38, 39],
            "marker_channel": 28,
            "name": "MindRoveSyncBox",
            "num_rows": 41,
            "other_channels": [19, 29, 30, 31],
            "package_num_channel": 26,
            "ppg_channels": [32, 33],
            "ppg_raw_channels": [34, 35, 36],
            "resistance_channels": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            "sampling_rate": 500,
            "timestamp_channel": 27,
        },
    },
}


def get_board_descr(board_id, preset=0):
    if board_id not in BOARD_DESCRIPTORS:
        raise ValueError(f"Unknown board_id: {board_id}")
    presets = BOARD_DESCRIPTORS[board_id]
    if preset not in presets:
        raise ValueError(f"Unknown preset {preset} for board_id {board_id}")
    return presets[preset]


def get_sampling_rate(board_id=0, preset=0):
    return get_board_descr(board_id, preset)["sampling_rate"]


def get_eeg_channels(board_id=0, preset=0):
    return get_board_descr(board_id, preset).get("eeg_channels", [])


def get_emg_channels(board_id=0, preset=0):
    return get_board_descr(board_id, preset).get("emg_channels", [])


def get_exg_channels(board_id=0, preset=0):
    return get_board_descr(board_id, preset).get("exg_channels", [])


def get_num_rows(board_id=0, preset=0):
    return get_board_descr(board_id, preset)["num_rows"]
