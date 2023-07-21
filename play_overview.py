import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def mirror_play(play, x_pos='x', target_dir='right', mirrored=False):

    x_max = 100
    if mirrored:
        play = play.with_columns((x_max-pl.col(x_pos)).alias(x_pos))

    elif (play['playDirection'].unique().item()!=target_dir):
        play = play.with_columns((x_max-pl.col(x_pos)).alias(x_pos))
        mirrored=True
        
    return play, mirrored


def make_scoreboard(play, ax):

    clock = play['gameClock'].item()
    home_score = play['preSnapHomeScore'].item()
    away_score = play['preSnapVisitorScore'].item()
    to_go = play['yardsToGo'].item()
    quarter = ...
    down = ...

    s = f'{clock}\n{home_score}-{away_score}\n{to_go} yds to go'
    ax.text(0, 60, s=s, fontweight=200, ha='center', va='center')


def make_field():

    fig, ax = plt.subplots(figsize=(8,4))
    x_range = (0,100)
    y_range = (0,53)
    [ax.vlines(x=yard_line, ymin=y_range[0], ymax=y_range[1], color='k', lw=1) 
    for yard_line in np.arange(105, step=10)]
    [ax.hlines(y=y, xmin=x_range[0], xmax=x_range[1], color='k', lw=1) for y in y_range]
    ax.set_yticks([])
    ax.fill_betweenx(y=y_range, x1=-10, x2=0, color='blue', alpha=0.5)
    ax.fill_betweenx(y=y_range, x1=100, x2=110, color='red', alpha=0.5)

    ax.set(xlim=(x_range[0]-20,x_range[1]+20), ylim=(y_range[0]-15,y_range[1]+15))
    return fig, ax


def next_first_down_line(play_summary, ax, mirror=False):

    line_of_scrimmage = play_summary['yardlineNumber'].item()
    if mirror:
        line_of_scrimmage = 100 - line_of_scrimmage
            
    yards_to_go = play_summary['yardsToGo'].item()

    first_down_line = line_of_scrimmage + yards_to_go
    ax.vlines(x=first_down_line, ymin=0, ymax=53, color='y', zorder=-1)

    return None


def label_completion(play_summary):

    outcome = play_summary['passResult'].item()
    outcome_symbols = {'C': ('Complete', 'g'), 'I':('Incomplete', 'r'),
                       'IN':('Interception', 'r')}
    s, color = outcome_symbols[outcome]

    return s, color


def possession_outcome():

    return None


def label_QB():

    return None


def label_target_receiver():

    return None