import numpy as np
import pandas as pd
from pathlib import Path

# --- sanity checks ---
def identifyTargetReceiver(week,plays,playWeek,weekId=None):
    if weekId is None: weekBool = np.full(len(playWeek), True)
    else: weekBool = playWeek==weekId
    # contains dataframe of relevant information for identifying targetReceiver on completed passing plays
    completions = plays.loc[(plays['passResult']=='C') & (weekBool),['gameId','playId','playDescription','possessionTeam','offensePlayResult']]
    numCompletions = len(completions)
    
    # these are the possible events: pass_arrived, pass_forward, pass_outcome_caught
    qbNflID = np.zeros(numCompletions)
    receiverDistance = np.zeros(numCompletions)
    receiverNextClosestDistance = np.zeros(numCompletions)
    receiverName = [None] * numCompletions
    descName = [None] * numCompletions
    for idx, gameIdPlayId in enumerate(zip(completions['gameId'],completions['playId'])):
        gameId, playId = gameIdPlayId
        playIdx = (week['gameId']==gameId) & (week['playId']==playId)
        currentPlay = week.loc[playIdx] # dataframe containing values from current play
        currentPlayDescription = plays.loc[(plays['gameId']==gameId) & (plays['playId']==playId), ['playDescription']].iloc[0].iloc[0]

        # Check if there's a valid QB (i.e. if only one nflId shows up at the QB position...)
        cID = np.unique(currentPlay.loc[currentPlay['position']=='QB','nflId'])

        # If so, add it to the register, otherwise, skip this play
        if len(cID)==1: qbNflID[idx] = cID[0]
        else: continue

        # Get which team is on offense (it's either "home" or "away", wtf)
        teamOffense = currentPlay.loc[currentPlay['nflId']==cID[0],'team'].iloc[0]

        # Find out row of frame when pass was caught
        event2use = 'pass_outcome_caught'
        footballCaughtPosition = currentPlay.loc[(currentPlay['team']=='football') & (currentPlay['event']==event2use)]
        if len(footballCaughtPosition)!=1: 
            # that means it's a touchdown
            event2use = 'pass_outcome_touchdown'
            footballCaughtPosition = currentPlay.loc[(currentPlay['team']=='football') & (currentPlay['event']==event2use)]
        if len(footballCaughtPosition)!=1: 
            # that means it's at the "pass_arrived" event
            event2use = 'pass_arrived'
            footballCaughtPosition = currentPlay.loc[(currentPlay['team']=='football') & (currentPlay['event']==event2use)]
        if len(footballCaughtPosition)!=1:
            raise ValueError("Couldn't figure out when the ball was caught on this play...")

        # Find out where the ball was caught
        catchLocation = [footballCaughtPosition['x'].iloc[0],footballCaughtPosition['y'].iloc[0]]

        # Find out where all offensive players are when the ball was caught
        offenseSnapshot = currentPlay.loc[(currentPlay['team']==teamOffense) & (currentPlay['event']==event2use)]
        offenseLocation = np.array(offenseSnapshot[['x','y']])
        distanceFromFootball = np.sqrt(np.sum((offenseLocation - catchLocation)**2,axis=1))
        idxDistanceFromFootball = np.argsort(distanceFromFootball)
        receiverDistance[idx] = distanceFromFootball[idxDistanceFromFootball[0]]
        receiverNextClosestDistance[idx] = distanceFromFootball[idxDistanceFromFootball[1]]
        receiverName[idx] = offenseSnapshot['displayName'].iloc[idxDistanceFromFootball[0]]

        # Get name of receiver from play description
        toIdx = currentPlayDescription.find(' to ')+4
        nextSpaceIdx = currentPlayDescription[toIdx:].find(' ')
        descName[idx] = currentPlayDescription[toIdx:toIdx+nextSpaceIdx]
    
    
    return receiverName, descName, receiverDistance, receiverNextClosestDistance, qbNflID

# -------------------------------------loading functions----------------------------------------
def dataPath(year='2021'):
    return Path(f'/Users/landauland/Documents/SportsScience/nfl-big-data-bowl-{year}')

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
    
def playWeek(games, plays):
    playWeek = np.zeros(len(plays))
    for game,week in zip(games['gameId'],games['week']):
        idxPlayDuringGame = plays['gameId']==game
        playWeek[idxPlayDuringGame] = week
    return playWeek