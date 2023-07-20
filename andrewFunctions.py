import socket
import numpy as np
import pandas as pd
import basicFunctions as bf
import matplotlib.pyplot as plt
from pathlib import Path

# Documentation of (some) of the data: https://www.kaggle.com/competitions/nfl-big-data-bowl-2021/data

# --- standard functions ---
def getCompletions(plays, playWeek, weekId):
    if weekId is None: weekBool = np.full(len(playWeek), True)
    else: weekBool = playWeek==weekId
    
    # contains dataframe of relevant information for identifying targetReceiver on completed passing plays
    completions = plays.loc[(plays['passResult']=='C') & (weekBool),['gameId','playId','playDescription','possessionTeam','offensePlayResult']]
    return completions

# --- sanity checks ---
def receiverVelocityCheck(week,plays,playWeek,weekId=None):
    """
    This function measures receiver velocity based on their x/y data.
    It also retrieves the speed/direction (s/dir) data, allowing for a comparison between the two values. 
    I just want to sanity check that this data makes sense and that I know how to interpret the values. 
    """
    
    # metaparameters:
    hfpd = 1
    
    # start by identifying the target receiver (don't need to sanity check every datum, reduce computation time)
    receiverName, validPlay = identifyTargetReceiver(week,plays,playWeek,weekId=weekId)[0:2]
    completions = getCompletions(plays, playWeek, weekId=weekId)
    
    assert len(receiverName)==len(completions), "receiverName and completions are not the same length"
    
    # clean up data
    receiverName = [rname for (rname,vplay) in zip(receiverName, validPlay) if vplay]
    completions = completions[validPlay]
    
    # go through each completion, pull out position & movement data from the target receiver
    dx,dy,speed,direction = [],[],[],[]
    for idx, gameIdPlayId in enumerate(zip(completions['gameId'],completions['playId'])):
        gameId, playId = gameIdPlayId
        
        # pull out target receivers data from the current play
        playIdx = (week['gameId']==gameId) & (week['playId']==playId) & (week['displayName']==receiverName[idx])
        receiverOnPlay = week.loc[playIdx] # dataframe containing values from current play
        receiverMovement = np.array(receiverOnPlay[['x','y','s','a','dis','o','dir']])
        
        # fivePointDer is a function that estimates the derivative (and crops the signal, so ifpd is the index)
        cdx,ifpd = bf.fivePointDer(receiverMovement[:,0],h=hfpd,returnIndex=True)
        dx.append(cdx)
        dy.append(bf.fivePointDer(receiverMovement[:,hfpd],h=1))
        speed.append(receiverMovement[ifpd,2])
        direction.append(receiverMovement[ifpd,6])
        
    dx,dy,speed,direction = np.concatenate(dx), np.concatenate(dy), np.concatenate(speed), np.concatenate(direction)
    
    spd = np.sqrt(dx**2 + dy**2)*10
    ang = np.arctan2(dx,dy)/2/np.pi*360
    error = np.mod(ang,360)-direction
    plt.scatter(np.mod(error+180,360)-180, speed, c='k', s=5, alpha=0.1)
    plt.xlabel('Error direction estimate (from data or from computed estimate from x/y movement)')
    plt.ylabel('Speed')
    plt.title('Error only high when speed is low!')
    return dx,dy,spd,ang,speed,direction
    
    

def identifyTargetReceiver(week,plays,playWeek,weekId=None):
    """
    This functions analyzes each play that results in a completion and identifies the target receiver on the play. 
    Although the playerName that received the catch is notated (w/ an annoying abbreviation) in the "plays" dataframe, 
    they aren't specifically mentioned in the tracking data, which means that a bit of a guess needs to be made. 
    (No guess needed if the abbreviated names and the display name have a one-to-one correspondence). 
    It does so by finding the location of the football at the time of catch, then choosing the offensive player
    that is closest to that position.
    """
    
    completions = getCompletions(plays, playWeek, weekId) # start by identify ing 
    numCompletions = len(completions)
    
    # create some arrays for saving the data about the pass reception
    event2use = ['pass_outcome_caught', 'pass_outcome_touchdown', 'pass_arrived']
    validPlay = np.full(numCompletions, False)
    qbNflID = np.zeros(numCompletions)
    receiverDistance = np.zeros(numCompletions)
    receiverNextClosestDistance = np.zeros(numCompletions)
    receiverName = [None] * numCompletions
    descName = [None] * numCompletions
    for idx, gameIdPlayId in enumerate(zip(completions['gameId'],completions['playId'])):
        gameId, playId = gameIdPlayId
        
        # get the current play description -- which contains an abbreviation of which player received the catch
        playIdx = (week['gameId']==gameId) & (week['playId']==playId)
        currentPlay = week.loc[playIdx] # dataframe containing values from current play
        currentPlayDescription = plays.loc[(plays['gameId']==gameId) & (plays['playId']==playId), ['playDescription']].iloc[0].iloc[0]
        
        # check if there is a single forward pass (otherwise it's a confusing and weird play... probably)
        validPlay[idx] = np.sum(currentPlay.loc[currentPlay['team']=='football']['event']=='pass_forward')==1
        if not(validPlay[idx]): continue
            
        # Check if there's a valid QB (i.e. if only one nflId shows up at the QB position...)
        cID = np.unique(currentPlay.loc[currentPlay['position']=='QB','nflId'])

        # If so, add it to the register, otherwise, skip this play
        if len(cID)==1: qbNflID[idx] = cID[0]
        else: continue

        # Get which team is on offense (it's either "home" or "away", rather than the city name from plays, wtf)
        teamOffense = currentPlay.loc[currentPlay['nflId']==cID[0],'team'].iloc[0]

        # Find out row of frame when pass was caught
        # these are the possible events: pass_arrived, pass_forward, pass_outcome_caught - I don't know what they all mean yet
        idxEvent = 0
        footballCaughtPosition = currentPlay.loc[(currentPlay['team']=='football') & (currentPlay['event']==event2use[0])]
        if len(footballCaughtPosition)!=1: 
            # that means it's a touchdown
            idxEvent = 1
            footballCaughtPosition = currentPlay.loc[(currentPlay['team']=='football') & (currentPlay['event']==event2use[1])]
        if len(footballCaughtPosition)!=1: 
            # that means it's at the "pass_arrived" event
            idxEvent = 2
            footballCaughtPosition = currentPlay.loc[(currentPlay['team']=='football') & (currentPlay['event']==event2use[2])]
        if len(footballCaughtPosition)!=1:
            raise ValueError("Couldn't figure out when the ball was caught on this play...")

        # Find out where the ball was caught
        catchLocation = [footballCaughtPosition['x'].iloc[0],footballCaughtPosition['y'].iloc[0]]

        # Find out where all offensive players are when the ball was caught
        offenseSnapshot = currentPlay.loc[(currentPlay['team']==teamOffense) & (currentPlay['event']==event2use[idxEvent])]
        offenseLocation = np.array(offenseSnapshot[['x','y']])
        distanceFromFootball = np.sqrt(np.sum((offenseLocation - catchLocation)**2,axis=1))
        idxDistanceFromFootball = np.argsort(distanceFromFootball)
        receiverDistance[idx] = distanceFromFootball[idxDistanceFromFootball[0]]
        receiverNextClosestDistance[idx] = distanceFromFootball[idxDistanceFromFootball[1]]
        receiverName[idx] = offenseSnapshot['displayName'].iloc[idxDistanceFromFootball[0]]

        # Get name of receiver from play description ("QB pass to ReceiverNameAbbreviated ...")
        toIdx = currentPlayDescription.find(' to ')+4
        nextSpaceIdx = currentPlayDescription[toIdx:].find(' ')
        descName[idx] = currentPlayDescription[toIdx:toIdx+nextSpaceIdx]
    
    return receiverName, validPlay, descName, receiverDistance, receiverNextClosestDistance, qbNflID

def checkAbbreviatedNames(plays):
    # Code for checking the abbreviated names!
    # Returns all unique abbreviated names (note that my detection algorithm isn't perfect)
    completions = plays.loc[(plays['passResult']=='C'),['playDescription']]
    numCompletions = len(completions)

    receiverAbbreviation = [None]*numCompletions
    for idxPlay, playDescription in enumerate(completions['playDescription']):
        passIdx = playDescription.find(' pass ')
        toIdx = playDescription[passIdx:].find(' to ')+passIdx+4
        nextSpaceIdx = playDescription[toIdx:].find(' ')
        receiverAbbreviation[idxPlay] = playDescription[toIdx:toIdx+nextSpaceIdx]    

    return np.unique(receiverAbbreviation)
        
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