# EMBRACE

## Usage
- `pip install -e pip`
- `python app.py`

## Additional Features
- Set env `EMBRACE_MRFZ=1` to see all-zero mock input without actual MindRove.
- Set env `EMBRACE_RES=1` to enable memory usage monitor (visible when you close the app).
- Port 8000 has a remote control page.

## Caveats
- Multithreading is misconfigured in some places. If the app behaves abnormally, restart it.
- Cannot guarantee that this is memory leak-free.