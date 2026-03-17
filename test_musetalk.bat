@echo off
set PYTHONHASHSEED=random
call conda activate MuseTalk
python -c "import sys; print('Python:', sys.executable)"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
