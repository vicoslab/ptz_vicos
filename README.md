#  AXIS 215 PTZ Resources

Manual: https://www.axis.com/dam/public/62/10/c9/axis-215ptz215ptz-e--users-manual-en-US-39697.pdf

Axis's Vapix library: https://github.com/derens99/vapix-python/tree/master

## Install
```bash
pip install vapix_python ultralytics
```
## Run demos
```bash
#WASD camera move, yolo detection
python3 test.py

#yolor person detection -> move camera to central pixel for tracking
python3 track_test.py
```
