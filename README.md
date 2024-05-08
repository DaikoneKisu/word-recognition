Before everything, download emnist letters datasets:
https://www.kaggle.com/datasets/crawford/emnist

Download files emnist-letters-test.csv and emnist-letters-train.csv and move them to word_recognition/assets folder so it looks like this:
word_recognition/assets/emnist-letters-test.csv
word_recognition/assets/emnist-letters-train.csv

Create and activate virtual environment:

```powershell
python -m venv .venv

On windows
```
.venv/Scripts/Activate.ps1
```

On ubuntu
```
source .venv/bin/activate
```

Install tkinter on ubuntu
```
apt-get install python-tk
apt-get install python3-tk
```

Install dependencies once inside venv:

```powershell
python -m pip install --editable .
```
