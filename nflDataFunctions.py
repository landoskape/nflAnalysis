import numpy as np
import pandas as pd
from pathlib import Path

def dataPath(year='2021'):
    return Path(f'/Users/landauland/Documents/SportsScience/nfl-big-data-bowl-{year}')

# -------------------------------------loading functions----------------------------------------
def loadGames(dpath=None):
    dpath = dataPath() if dpath is None else dpath
    return pd.read_csv(dpath/'games.csv')

def loadPlayers(dpath=None):
    dpath = dataPath() if dpath is None else dpath
    return pd.read_csv(dpath/'players.csv')

def loadPlays(dpath=None):
    dpath = dataPath() if dpath is None else dpath
    return pd.read_csv(dpath/'plays.csv')

def loadPffData(dpath=None):
    dpath = dataPath() if dpath is None else dpath
    if (dpath/'pffScoutingData.csv').exists():
        return pd.read_csv(dpath/'pffScoutingData.csv')
    return None

def loadData(dpath=None):
    return loadGames(dpath), loadPlayers(dpath), loadPlays(dpath), loadPffData(dpath)

def loadWeek(dpath=None):
    return tuple([loadWeekData(week=week) for week in range(1,9)])

def loadWeekData(dpath=None, week=None):
    assert isinstance(week, int), "week must be an integer between 1 and 8!"
    dpath = dataPath() if dpath is None else dpath
    return pd.read_csv(dpath/f'week{week}.csv')
    
    