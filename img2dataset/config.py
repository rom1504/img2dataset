# Used to store package wide constants
import os

if os.path.isdir(os.path.expanduser("~/.img2dataset")):
    DB_FILE = os.path.expanduser("~/.img2dataset/img2dataset.db")
elif os.getenv("XDG_DATA_HOME"):
    DB_FILE = os.path.join(os.getenv("XDG_DATA_HOME"), "img2dataset", "img2dataset.db")
elif os.getenv("APPDATA"):
    DB_FILE = os.path.join(os.getenv("APPDATA"), "img2dataset", "img2dataset.db")
else:
    DB_FILE = os.path.join(os.getenv("HOME"), ".local", "share", "img2dataset", "img2dataset.db")

# Global variable used to keep track of what is downloading
currently_downloading = []