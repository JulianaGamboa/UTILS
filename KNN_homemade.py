
#Euclidean distance
import numpy as np

def distance (p1, p2):
    """Find the distance between two points."""
    return np.sqrt(np.sum(np.power(p2 - p1, 2)))

p1 = np.array ([1,1])
p2 = np.array ([4,4])
distance (p1, p2)

import random

def majority_vote (votes):
    """Return the most common element in votes."""
    vote_counts = {}
    for vote in votes:
        if vote in vote_counts:
            vote_counts[vote] +=1
        else:
            vote_counts[vote] = 1        
    winners = []
    max_count = max(vote_counts.values())
    for vote, count in vote_counts.items():
        if count == max_count:
            winners.append(vote)
    return random.choice(winners) #en caso de empate va a elegir un winner al azar

votes = [2,3,3,3,1,3,3,2,2,2,2]
winner = majority_vote(votes)
winner 

#max(vote_counts.keys())  
#max(vote_counts.values())  
#max_counts = max(vote_counts.values())  #lo extraigo como variable

import scipy.stats as ss
def majority_vote_short (votes):
    """Return the most common element in votes."""
    mode, count = ss.mstats.mode(votes) #utiliza la moda
    return mode

votes = [2,3,3,3,1,3,3,2,2,2,2]
winner = majority_vote_short(votes)
winner

















