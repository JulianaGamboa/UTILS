
import numpy as np

points = np.array([[1,1], [1,2], [1,3], [2,1], [2,2], [2,3], [3,1], [3,2], [3,3]])
p = np.array ([2.5, 2])

import matplotlib.pyplot as plt
plt.plot(points[:,0], points[:,1], "ro")

plt.plot (p[0], p[1], "bo")

def distance (p1, p2):
    """Find the distance between two points."""
    return np.sqrt(np.sum(np.power(p2 - p1, 2)))

def find_nearest_neighbors(p, points, k=5):
    """Find the k nearest neighbors of point p and return their indices."""
    distances = np.zeros(points.shape[0])
    for i in range (len(distances)):
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances) #it returns to indices that would sort the given array (de menor a mayor)
    return ind[:k]

ind = find_nearest_neighbors(p, points, 2); print (points[ind])
ind = find_nearest_neighbors(p, points, 3); print (points[ind])
ind = find_nearest_neighbors(p, points, 4); print (points[ind])
ind = find_nearest_neighbors(p, points, 5); print (points[ind])

# distances
# distances[4] #dice la distancia
# points[4] #dice la ubicación

# distances[ind] #it returns distances values sorted (from < to >)
# distances [ind[0:2]] #it returns the 2 nearest distances

#########################################################
######  FUNCIÓN PARA PREDECIR LA CLASE DEL PUNTO P ######
#########################################################

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

def knn_predict (p, points, outcomes, k=5):
    ind = find_nearest_neighbors(p, points, k)
    return majority_vote(outcomes[ind]) #outcomes es la matriz de clasificación

outcomes = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1]) #tiene que tener la misma longitud que points
len (outcomes)

knn_predict(np.array([2.0,2.5]), points, outcomes, k=3)

#### 3.3.5. GENERATING SYNTHETIC DATA #####

from scipy.stats import norm
def generate_synth_data(n=50):
    """Create two sets of point from bivariate normal distributions."""
    points=np.concatenate((norm(0,1).rvs((n,2)), norm(1,1).rvs((n,2))), axis=0)
    outcomes = np.concatenate((np.repeat(0,n), np.repeat(1,n)))
    return (points, outcomes)

n=20
plt.figure ()
plt.plot(points [:n,0], points [:n, 1], "ro")
plt.plot(points [n:,0], points [n:, 1], "bo")
plt.savefig ("bivardata.pdf")

#### 3.3.6. MAKING A PREDICTION GRID ####
#########################################

def make_prediction_grid (predictors, outcomes, limits, h, k):
    """Classify each point on the prediction grid."""
    (x_min, x_max, y_min, y_max) = limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs, ys)
    
    prediction_grid = np.zeros(xx.shape, dtype = int)
    for i,x in enumerate(xs):
        for j,y in enumerate (ys):
            p= np.array([x,y])
            prediction_grid[i,j] = knn_predict(p, predictors, outcomes, k)
    return (xx, yy, prediction_grid)

####3.3.7. PLOTTING THE PREDICTION GRID ####
############################################

def plot_prediction_grid (xx, yy, prediction_grid, filename):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    plt.savefig(filename)

(predictors, outcomes) = generate_synth_data()

k=5; filename = "knn_synth_5"; limits = (-3, 4, -3, 4); h=0.1
(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, filename)

k=10; filename = "knn_synth_50"; limits = (-3, 4, -3, 4); h=0.1
(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, filename)


