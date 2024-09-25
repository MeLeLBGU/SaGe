import os
from pathlib import Path


PATH_SAGE = Path(os.getcwd())
def setSageFolder(path: Path):
    global PATH_SAGE
    PATH_SAGE = path

def getDataFolder() -> Path:
    path = PATH_SAGE / "data"
    path.mkdir(exist_ok=True)
    return path

def getResultsFolder() -> Path:
    path = PATH_SAGE / "results"
    path.mkdir(exist_ok=True)
    return path

def getLogsFolder() -> Path:
    path = PATH_SAGE / "logs"
    path.mkdir(exist_ok=True)
    return path
