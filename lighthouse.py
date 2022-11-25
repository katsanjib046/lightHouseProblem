# Based on code  translated to Python by Issac Trotts in 2007
#  Lighthouse at (x,y,z) emitted n flashes observed at positions on coast.
# Inputs:
#  Prior(u)    is uniform (=1) over (0,1), mapped to x = 4*u - 2; and
#
#  Prior(w)    is uniform (=1) over (0,1), mapped to z = 2*w; so that
#  Position    is 2-dimensional -2 < x < 2, 0 < z < 2 with flat prior
#  Likelihood  is L(x,z) = PRODUCT[k] (z/pi) / ((D[k] - x)^2 + z^2)
# Outputs:
#  Evidence    is Z = INTEGRAL L(x,z) Prior(x,z) dxdz
#  Posterior   is P(x,z) = L(x,z) / Z estimating lighthouse position
#  Information is H = INTEGRAL P(x,z) log(P(x,z)/Prior(x,z)) dxdz

import numpy as np
import matplotlib.pyplot as plt
import sys
from mininest import nested_sampling
from sklearn.cluster import KMeans
from KDEpy import TreeKDE

# plot style
plt.style.use('dark_background')



def main():
    args = sys.argv
    global dim
    global num_lighthouse
    if len(args) == 1:
        print("Usage: python lighthouse.py num_lighthouse(optional) dim(optional)")
        print("Default case: num_lighthouse=1 and dim=2 will be used.")
        num_lighthouse=1
        dim=2
    elif len(args) == 3:
        try:
            num_lighthouse = int(args[1])
            dim = int(args[2])
            if dim not in [2,3]:
                print("dim must be 2 or 3")
                sys.exit()
        except ValueError:
            print("Usage: python lighthouse.py num_lighthouse(optional) dim(optional)")
    else:
        sys.exit("Usage: python lighthouse.py num_lighthouse(optional) dim(optional)")

    # number of flashes. Default is 1000.
    samples_for_eachLH = 1000
    # generate lighthouse coordinates
    global lightHCoords 
    lightHCoords= generateLighthouse(num_lighthouse, dim)
    print("Actual Lighthouse coordinates: ", lightHCoords)

    # generate flashes positions
    global X
    global Y
    X, Y = generatePositions(lightHCoords, samples_for_eachLH, dim)
    # visualize the flashes
    visualizeFlashes(X, Y, dim)

    # run nested sampling
    num_object = 100
    max_iter = 10000
    results = nested_sampling(num_object, max_iter, sample_from_prior, explore)

    posteriors, weights, clusterCenterPositions, kmeans, statData = process_results(results)
    do_plots(posteriors, clusterCenterPositions, weights)
    plt.show()

# conversion from u or w to x or z
transverse = lambda unit: 4.0 * unit - 2.0 
height = lambda unit: 2.0 * unit       


def generateLighthouse(num_lighthouse=1, dim=2):
    """
    Description:
        Randomly generates coordinates of lighthouses.
    Args:
        num_lighthouse: The number of lighthouses.
        dim: Dimension of the problem.
    Returns:
        lightHCoords: The coordinates of lighthouses.
    Note:
        lightHCoords=([[1st],[2nd]]) for 2LH.
        lightHCoords=([[1st]]) for 1LH.
        The values of x, y and z (of 3D) are randomly generated but are within in the range of [-2,2] or [0,2].
    """
    lightHCoords=[]
    for i in range(num_lighthouse):
        if dim==2:
            lightHCoords.append([np.random.uniform(-2,2),np.random.uniform(0,2)])
        elif dim==3:
            lightHCoords.append([np.random.uniform(-2,2),np.random.uniform(-2,2),np.random.uniform(0,2)])
    return lightHCoords


def generatePositions(lightHCoords, samples_for_eachLH=1000, dim=2):
    
    """
    Description:
        Randomly generates a 'theta' and 'phi' as numpy arrays.
    Args:
        lightHCoords: A numpy array containing LH coordinates in 2D/3D.
        samples_for_eachLH: The number of flashes.
        dim: Dimension of the problem.
    Returns:
        (X, Y): The position of flashes observed at the shore.
    
    Note: 
        lightHCoords=([[1st],[2nd]]) for 2LH.
        lightHCoords=([[1st]]) for 1LH.
    """
    X=[]
    Y=[]
    for i in range(len(lightHCoords)):
        x=lightHCoords[i][0]
        z=lightHCoords[i][-1]
        if dim==2:
            thetaArray = np.random.uniform(-np.pi/2,np.pi/2,samples_for_eachLH)
            flashesPositionsX, flashesPositionsY = z * np.tan(thetaArray) + x ,\
            np.zeros(samples_for_eachLH)
        elif dim==3:
            y=lightHCoords[i][1]
            thetaArray = np.random.uniform(0,np.pi/2,samples_for_eachLH)

            flashesPositionsX, flashesPositionsY = z * np.tan(thetaArray), np.zeros(samples_for_eachLH)

            phiArray = np.random.uniform(0,2*np.pi,samples_for_eachLH)
            flashesPositionsX, flashesPositionsY = x + np.cos(phiArray)*(flashesPositionsX) - np.sin(phiArray)*(flashesPositionsY),\
                                                   y + np.sin(phiArray)*(flashesPositionsX) + np.cos(phiArray)*(flashesPositionsY)
        X,Y=np.append(X,[flashesPositionsX]),np.append(Y,[flashesPositionsY])


    return (X,Y)

def visualizeFlashes(X,Y,dim=2):
    """
    Description:
        Visualize the flashes observed at the shore.
    Args:
        X: A numpy array containing the x coordinates of flashes.
        Y: A numpy array containing the y coordinates of flashes.
    Returns:
        None
    """
    plt.figure('Flashes (Data)')
    plt.title("Distribution of flashes")
    if dim==2:
        plt.hist(X,bins=50, range=(-10,10))
        plt.xlabel("x")
        plt.ylabel("Number of flashes")
        # plt.ylim(0,800)
        plt.xlim(-10,10)
    elif dim==3:
        plt.plot(X,Y,'.')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(-10,10)
        plt.ylim(-10,10)
    plt.show()

#------------------ Class for Light House ------------------#
class LHouses():

    """
    Class definition for collection of lighthouses.
    """
    
    def __init__(self,unitArray):
        """
        Initializes the class with the following attributes.
        Note: For 2 lighthouses in 3D, unitArray should be a (2,3) array.
        """
        configDim = np.size(unitArray)
        assert(configDim%dim==0 and configDim>1)
        self.update(unitArray)
        self.logWt=None     # log(Weight), adding to SUM(Wt) = Evidence Z

    def update(self,unitArray):
        """
        Creates a new instance of the coordinate value.
        Computes the loglikehood of the coordinate.
        """
        configDim = np.size(unitArray)
        assert(configDim % dim == 0 and configDim>1)
        self.unitCoords = np.zeros(unitArray.shape)
        for indexTuple , unitSample in np.ndenumerate(unitArray):
            self.unitCoords[indexTuple] = unitSample  # Uniform-prior controlling parameter for position
        self.mapUnitToXYZ()
        self.assignlogL()

    def mapUnitToXYZ(self):
        """
        Converts from unit coordinates to lighthouse position(s)
        """
        self.Coords = np.zeros(self.unitCoords.shape)
        for indexTuple , unitSample in np.ndenumerate(self.unitCoords):
            if indexTuple[-1] != dim-1:
                self.Coords[indexTuple] = transverse(unitSample)
            else:
                self.Coords[indexTuple] = height(unitSample)

    def assignlogL(self):
        """
        Assigns the attribute logLikelihood = ln Prob(data | position)
        """
        self.logL = logLhoodLHouse(self.Coords)

    def copy(self):
        """
        Returns the copy of the instance
        """
        return LHouses(self.unitCoords)


def logLhoodLHouse(lightHCoords):
    
    """
    Args:
        lightHCoords: Contains the coordinates of a lighthouse.
    Returns:
        logL: The log likelihood value for the given argument.
    Description:
        Uses specific formula for 2D and 3D case to calculate likelihood.
    """

    x = np.array( lightHCoords[...,0])
    z = np.array(lightHCoords[...,-1])
    DX = X
    sumLikelihoodLH = 0

    if dim ==2:
        if np.sum(x.shape) == 0:
            sumLikelihoodLH = (z / np.pi) / ((DX - x)*(DX - x) + z*z)
        else:
            for e in range(num_lighthouse):
                sumLikelihoodLH += (1/num_lighthouse)* (z[e] / np.pi) / ((DX - x[e])*(DX - x[e]) + z[e]*z[e])


    elif dim==3:
        y = np.array(lightHCoords[...,1])
        DY = Y
        if np.sum(x.shape) == 0:
            sumLikelihoodLH = (z / np.pi**2) / ((DX - x)*(DX - x) + (DY - y)*(DY - y) + z*z) / np.sqrt((DX - x)*(DX - x) + (DY - y)*(DY - y))
        else:
            for e in range(num_lighthouse):
                sumLikelihoodLH += (1/num_lighthouse)* (z[e] / np.pi**2) / ((DX - x[e])*(DX - x[e]) + (DY - y[e])*(DY - y[e]) + z[e]*z[e]) / np.sqrt((DX - x[e])*(DX - x[e]) + (DY - y[e])*(DY - y[e]))

    logL = np.sum( np.log(sumLikelihoodLH ))
    return logL

def sample_from_prior():
    
    """
    Args:
        None.
    Returns:
        Obj: An object of the LHouses class.
    Description:
        Generates a 2D/3D coordinate and creates an object.
    """
    
    unitCoords = np.random.uniform(size=(num_lighthouse,dim))
    unitCoords = np.squeeze(unitCoords) # if (1,dim) squeeze to (dim,)
    Obj = LHouses(unitCoords)
    return Obj

def explore(Obj,logLstar):
   
    """
    Args:
        Obj: An instance of the LHouses class.
        logLstar: The least likelihood value used in sampling.
    Returns:
        ret: A modified version of Obj.
    Description:        
        Performs Markov Chain Monte Carlo (MCMC) to modify the original object.  
        Object is evolved with likelihood constraint L > Lstar.
    """
    
    ret =  Obj.copy() 
    step = 0.1   # Initial guess suitable step-size in (0,1)
    accept = 0   # # MCMC acceptances
    reject = 0   # # MCMC rejections
    a = 1.0
    Try = Obj.copy()     # Trial object
    for _ in range(20):  # pre-judged number of steps

        # Trial object u-w step
        unitCoords_New = ret.unitCoords + step * (2.0*np.random.uniform(size=ret.unitCoords.shape) - 1.0)  # |move| < step
        unitCoords_New -= np.floor(unitCoords_New)      # wraparound to stay within (0,1)
        Try.update(unitCoords_New)

        # Accept if and only if within hard likelihood constraint
        if Try.logL > logLstar:
            ret = Try.copy()
            accept+=1
        else:
            reject+=1

        # Refine step-size to let acceptance ratio converge around 50%
        if( accept > reject ):
            step *= np.exp(a / accept)
            a /= 1.5
        if( accept < reject ):
            step /= np.exp(a / reject)
            a *= 1.5
    return ret

def get_posteriors(results):
    
    """
    Args:
        results: A dictionary data returned by the mininest function.
    Returns:
        posteriors: A numpy array containing x,y,z coordinates.
    Description:
        Determines the dimension of the array required for posterioirs.
        Extracts coordinate from results and appends them to posteriors.
    """
    
    ni = results['num_iterations']
    samples = results['samples']
    shape =  samples[0].Coords.shape
    posteriors = np.zeros(sum( ( shape, (ni,) ), () ) )
    for i in range(ni):
        coords = samples[i].Coords
        posteriors[...,i] = coords
    posteriors = np.swapaxes(posteriors, 0, -2)
    posteriors = posteriors.reshape((dim,num_lighthouse*ni))
    return posteriors

def get_statistics(results,weights=None):
    
    """
    Args:
        results: A dictionary data returned by the mininest function.
        weights: A numpy array containing weight distribution for posterior data samples.
    Returns:
        statData: A list of tuples containing statistical data.
    Description:
        Extracts the mean and standard deviation from results.
        Prints the extracted data.
    """
    
    ni = results['num_iterations']
    samples = results['samples']
    shape =  samples[0].Coords.shape
    avgCoords = np.zeros(shape) # first moments of coordinates
    sqrCoords = np.zeros(shape) # second moments of coordinates
    logZ = results['logZ']
    for i in range(ni):
        coords = samples[i].Coords
        avgCoords += weights[i] * coords
        sqrCoords += weights[i] * coords * coords
        
    meanX, sigmaX = avgCoords[0], np.sqrt(sqrCoords[0]-avgCoords[0]*avgCoords[0])
    print("\nmean(x) = %f, stddev(x) = %f" %(meanX, sigmaX))

    if dim==3:
        meanY, sigmaY = avgCoords[1], np.sqrt(sqrCoords[1]-avgCoords[1]*avgCoords[1])
        print("mean(y) = %f, stddev(y) = %f" %(meanY, sigmaY))

    meanZ, sigmaZ = avgCoords[-1], np.sqrt(sqrCoords[-1]-avgCoords[-1]*avgCoords[-1])
    print("mean(z) = %f, stddev(z) = %f" %(meanZ, sigmaZ))
    logZ_sdev = results['logZ_sdev']
    print("Evidence: ln(Z) = %g +- %g"%(logZ,logZ_sdev))

    # Analyze the changes in x,y,z and evidence for different z values
    statData = []
    statData.append((meanX, sigmaX))
    if dim==3: statData.append((meanY, sigmaY))
    statData.append((meanZ, sigmaZ))
    statData.append((logZ, logZ_sdev))
    return statData

def clustering(posteriors,weights=None,extraClusters=20):
    
    """
    Args:
        posteriors: A numpy array containing posterior coordinates.
        weights: A numpy array containing weight distribution for posterior data samples.
        extraClusters: An average calculator used for higher accuracy.
    Returns:
        clusterCenterPositions: The mean value of the estimated LH coordinate.
        kmeans: An object instance that fits the data according to the cluster.
    Description:
        Required for multiple lighthouses.
        Determines LH positions by finely differentiating the posterior values.
        Performs clustering 20 times to achieve better estimate.   
    """
    
    posteriorPoints = posteriors.T
    kmeans = KMeans(n_clusters=num_lighthouse,max_iter=1000,tol=1E-7,n_init=100).fit(posteriorPoints,weights)
    clusterCenterPositions = kmeans.cluster_centers_
    kmeans2 = KMeans(n_clusters=num_lighthouse+extraClusters,max_iter=1000,tol=1E-7,n_init=100).fit(posteriorPoints,weights)
    clusterCenterPositions2 = kmeans2.cluster_centers_
    # print(clusterCenterPositions2)
    for i in range(len(clusterCenterPositions[...,0])):
        idx = np.argmin(np.sum(np.abs(clusterCenterPositions[i] - clusterCenterPositions2),axis=1))
        clusterCenterPositions[i] = clusterCenterPositions2[idx]

    print("Cluster positions:")
    print(clusterCenterPositions)
    return clusterCenterPositions , kmeans

def process_results(results):
    
    """
    Args:
        results: A dictionary data returned by the mininest function.
    Returns:
        posteriors: A numpy array containing posterior coordinates.
        weights: A numpy array containing weight distribution for posterior data samples.
        clusterCenterPositions: The mean value of the estimated LH coordinate.
        kmeans: An object instance that fits the data according to the cluster.
        statData: A list of tuples containing statistical data.
    Description:
        Serves as a hub for the main function.
    """
    
    posteriors = get_posteriors(results)
    weights = get_weights(results,num_lighthouse)
    clusterCenterPositions , kmeans = clustering(posteriors,weights)
    if len(lightHCoords)==1:
        statData = get_statistics(results,weights)
    else:
        statData = None
    return posteriors, weights, clusterCenterPositions, kmeans, statData


def get_weights(results, num_lighthouse):
    
    """
    Args:
        results: A dictionary data returned by the mininest function.
    Returns:
        weights: A numpy array containing weight distribution in the posterior data samples.
    Description:
        Extracts the evidence values from results.
    """
    
    ni = results['num_iterations']
    samples = results['samples']
    logZ = results['logZ']
    weights = [0]*ni
    for i in range(ni):
        weights[i] = np.exp(samples[i].logWt - logZ)
    weights = weights * num_lighthouse
    weights = np.array(weights)

    return weights


def plot_weights(weights, model_num_LH):
        
    """
    Args:
        weights: A numpy array containing weight distribution for posterior data samples.
    Returns:
        None.
    Description:
        Plots the weight distribution vs number of iteration.
    """
    
    plt.figure('Weights')
    plt.title("Weights distribution")
    plt.xlabel('Number of iterations')
    plt.ylabel('Weights')

    plt.plot(weights[:len(weights)//model_num_LH])

def do_plots(posteriors, clusterCenterPositions, weights):
    
    """
    Args:
        posteriors: A numpy array containing posterior coordinates.
        weights: A numpy array containing weight distribution for posterior data samples.
    Returns:
        None
    Description:
        Plot the weight distribution.
        Plot the 3D graph for posterior for 3D case.
        Plot the cornerplot to show posterior data.
    """
    
    print("\nGenerating Plots. This might take some time...")
    plot_weights(weights, num_lighthouse)
    if dim==3: threeDimPlot(posteriors,clusterCenterPositions,weights)
    cornerplots(posteriors, weights)

def threeDimPlot(posteriors,clusterCenterPositions,weights=None):
    
    """
    Args:
        posteriors: A numpy array containing posterior coordinates.
        weights: A numpy array containing weight distribution for posterior data samples.
    Returns:
        None.
    Description:
        Plots the actual LH coordinate and the estimated LH coordinates in a 3D box. 
    """
    
    fig = plt.figure('{}-D plot'.format(dim))
    ax = fig.add_subplot(111, projection='3d')
    xp, yp, zp = posteriors[0,:],posteriors[1,:],posteriors[2,:]
    xyz = np.vstack([xp,yp,zp]).T
    kde = TreeKDE(kernel='gaussian', norm=2,bw=0.05)
    color = kde.fit(xyz,weights).evaluate(xyz)
    ax = plt.gca()
    scatter = ax.scatter(xs=xp, ys=yp, zs=zp, c=color,cmap="hot")
    plt.colorbar(scatter)

    actual = np.array(lightHCoords)
    
    for i in range(len(lightHCoords)):
        for j in range(3):
            if j!=0:
                xA = [lightHCoords[i][0], lightHCoords[i][0]]
                xC = [clusterCenterPositions[i][0],clusterCenterPositions[i][0]]
            else:
                xA = xC = [-2,2]
            if j!=1:
                yA = [lightHCoords[i][1], lightHCoords[i][1]]
                yC = [clusterCenterPositions[i][1],clusterCenterPositions[i][1]]
            else:
                yA = yC = [-2,2]
            if j!=2:
                zA = [lightHCoords[i][2], lightHCoords[i][2]]
                zC = [clusterCenterPositions[i][2],clusterCenterPositions[i][2]]
            else:
                zA = zC = [0, 2]
            ax.plot(xA,yA,zA,'r--',alpha=0.8, linewidth=3)
            ax.plot(xC,yC,zC,'g--',alpha=0.8, linewidth=3)
    ax.scatter(xs=actual[...,0],ys=actual[...,1],zs=actual[...,2],marker = '*',color='red',s=200,depthshade=False,label='Actual LH')
    x , y , z = [] , [] , []
    for i in range(num_lighthouse):
        x.append(clusterCenterPositions[i][0])
        y.append(clusterCenterPositions[i][1])
        z.append(clusterCenterPositions[i][-1])
    ax.scatter(x,y,z,marker = '*',color='green',s=200,depthshade=False,label='Cluster Estimate')
    ax.set_xlim(-2,2),ax.set_ylim(-2,2),ax.set_zlim(0,2)
    ax.set_xlabel('X axis'),ax.set_ylabel('Y axis'),ax.set_zlabel('Z axis')
    ax.set_title('A 3D-Plot of posterior points',weight='bold',size=12)
    plt.legend()
    plt.tight_layout()

def cornerplots(posteriors,weights=None):

    """
    Args:
        posteriors: A numpy array containing posterior coordinates.
        weights: A numpy array containing weight distribution for posterior data samples.
    Returns:
        None.
    Description:
        Plots individually the posterior data for x, y and z.
        Creates a histogram plot and a scatter plot estimating the LH coordinates.
    """
    
    pSize = posteriors[...,0].size # total number of posterior coordinates (3 for a single lhouse)
    numLhouses = pSize//dim
    transverseDomain = (-2,2)
    depthDomain = (0,2)
    domains = sum( ((transverseDomain,)*(dim-1),(depthDomain,))*numLhouses, () )
    plt.figure("Posterior plots")
    plt.title("Posterior distribution of lighthouse(s)")
    for i in range(pSize):
        plt.subplot(pSize,pSize,i*pSize+i+1)
        samples = posteriors[i]
        x = np.linspace(*domains[i],2000)
        estimator = TreeKDE(kernel='gaussian', bw=0.01)
        y = estimator.fit(samples, weights=weights).evaluate(x)
        plt.plot(x, y)
        try:
            plt.hist(samples,bins=50,range = domains[i],weights=weights,density=True)
        except AttributeError:
            plt.hist(samples,bins=50,range = domains[i],weights=weights,normed=True)
        if i==0:
            plt.title("X Posterior Data")
            plt.axvline(x=lightHCoords[0][0], color='r', linestyle='dashed')
            for k in range(len(lightHCoords)):
                plt.axvline(x=lightHCoords[k][0], color='r', linestyle='dashed')
        elif i==1:
            plt.title("Y Posterior Data")
            plt.axvline(x=lightHCoords[0][1], color='r', linestyle='dashed')
            for k in range(len(lightHCoords)):
                plt.axvline(x=lightHCoords[k][1], color='r', linestyle='dashed')
        else:
            plt.title("Z Posterior Data")
            plt.axvline(x=lightHCoords[0][2], color='r', linestyle='dashed')
            for k in range(len(lightHCoords)):
                plt.axvline(x=lightHCoords[k][2], color='r', linestyle='dashed') 
        
        # Joint posteriors
        for j in range(i):
            subPltIndex = i*pSize + 1 + j
            plt.subplot(pSize,pSize,subPltIndex)
            xp, yp = posteriors[j],posteriors[i]
            xy = np.vstack([xp,yp]).T
            kde = TreeKDE(kernel='gaussian', norm=2,bw=0.05)
            grid, points = kde.fit(xy,weights).evaluate(2**8)
            
            # The grid is of shape (obs, dims), points are of shape (obs, 1)
            x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
            z = points.reshape(2**8, 2**8).T
            
            # Plot the kernel density estimate
            ax = plt.gca()
            ax.contourf(x, y, z, 1000, cmap="hot")
            plt.xlim(domains[j])
            plt.ylim(domains[i])
            if i==1:
                plt.ylabel('y')
            else:
                if j==0:
                    plt.xlabel('x')
                    plt.ylabel('z')
                else:
                    plt.xlabel('y')
    plt.tight_layout()

#----------------- Testing and Running -----------------#
if __name__ == '__main__':
    main()
    