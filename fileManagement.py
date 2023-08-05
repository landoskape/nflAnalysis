import socket
from pathlib import Path
import numpy as np
import polars as pl

# -------------------------------------loading functions----------------------------------------
def dataPath(year='2021'):
    # function to return path to csv files for particular year of the competition
    hostName = socket.gethostname()
    if 'Andrews-MBP' in hostName:
        return Path(f'/Users/landauland/Documents/SportsScience/nfl-big-data-bowl-{year}')
    elif 'Celia' in hostName:
        raise ValueError(f"Have not coded the path location on {hostname} yet! - To do so, edit the dataPath() function")
        #return Path(f'/path/to/wherever/you/keep/the/data/nfl-big-data-bowl-{year}')
    else:
        raise ValueError(f"Did not recognize hostname ({hostName})")

def loadGames(dpath=None):
    dpath = dataPath() if dpath is None else dpath
    return pl.read_csv(dpath/'games.csv')

def loadPlayers(dpath=None):
    dpath = dataPath() if dpath is None else dpath
    return pl.read_csv(dpath/'players.csv')

def loadPlays(dpath=None):
    dpath = dataPath() if dpath is None else dpath
    return pl.read_csv(dpath/'plays.csv')

def loadPffData(dpath=None):
    dpath = dataPath() if dpath is None else dpath
    if (dpath/'pffScoutingData.csv').exists():
        return pl.read_csv(dpath/'pffScoutingData.csv')
    return None

def loadData(dpath=None):
    return loadGames(dpath), loadPlayers(dpath), loadPlays(dpath), loadPffData(dpath)

def loadWeek(dpath=None):
    return tuple([loadWeekData(week=week) for week in range(1,9)])

def loadWeekData(dpath=None, week=None):
    assert isinstance(week, int), "week must be an integer between 1 and 8!"
    dpath = dataPath() if dpath is None else dpath
    return pl.read_csv(dpath/f'week{week}.csv')
    
def playWeek(games, plays):
    playWeek = np.zeros(len(plays))
    for game,week in zip(games['gameId'],games['week']):
        idxPlayDuringGame = np.array(plays['gameId']==game).astype(bool)
        playWeek[idxPlayDuringGame] = week
    return playWeek


















