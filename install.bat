call venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
pip uninstall -y opencv-python
pip install opencv-contrib-python==4.6.0.66