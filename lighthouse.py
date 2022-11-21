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
import random
from mininest import nested_sampling

# plot style
plt.style.use('dark_background')



def main():
    args = sys.argv
    global dim
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

    num_lighthouse = len(lightHCoords)
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
    process_results(results)
    weights = get_weights(results, num_lighthouse)
    plot_weights(weights, num_lighthouse)

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
class LightHouse:
    """
    Definition for collection of light houses.
    """
    def __init__(self):
        """
        Description:
            Initialize a light house.
        Returns:
            None
        """
        self.u = None
        # self.v = None
        self.w = None
        self.x = None
        # self.y = None
        self.z = None
        self.logL = None
        self.logWt = None

# --------------------------------------------

def sample_from_prior():
    """
    Description:
        Sample from the prior distribution.
    Returns:
        Obj: An instance of the LightHouse class. 
    """
    Obj = LightHouse()
    Obj.u = random.random()                # uniform in (0,1)
    Obj.w = random.random()                # uniform in (0,1)
    Obj.x = transverse(Obj.u)             # map to x
    Obj.z = height(Obj.w)                    # map to y
    Obj.logL = logLikeHood([[Obj.x, Obj.z]], dim)
    return Obj

def logLikeHood(LHCoords, dim=2):
    """
    Description:
        Calculate the log likelihood of the flashes.
    Args:
        lightHCoords: A numpy array containing LH coordinates in 2D/3D.
        X: A numpy array containing the x coordinates of flashes.
        Y: A numpy array containing the y coordinates of flashes.
        dim: Dimension of the problem.
    Returns:
        logLikeHood: The log likelihood of the flashes.
    """
    logLikeHood=0
    for i in range(len(LHCoords)):
        x,z=LHCoords[i][0],LHCoords[i][-1]
        if dim==2:
            for e in range(len(X)):
                logLikeHood += np.log((z / np.pi) / ((X[e] - x)*(X[e] - x) + z*z))
        elif dim==3:
            y = lightHCoords[i][1]
            for e in range(len(X)):
                logLikeHood += np.log((z / np.pi**2) / ((X[e] - x)*(X[e] - x) + (Y[e] - y)*(Y[e] - y) + z*z) / np.sqrt((X[e] - x)*(X[e] - x) + (Y[e] - y)*(Y[e] - y)))
            
    return logLikeHood


def explore(Obj, logLStar):
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
    ret = LightHouse()
    ret.__dict__ = Obj.__dict__.copy()
    step = 0.1;   # Initial guess suitable step-size in (0,1)
    accept = 0;   # # MCMC acceptances
    reject = 0;   # # MCMC rejections
    Try = LightHouse();          # Trial object

    for m in range(20):  # pre-judged number of steps

        # Trial object
        Try.u = ret.u + step * (2.*random.random() - 1.);  # |move| < step
        Try.w = ret.w + step * (2.*random.random() - 1.);  # |move| < step
        Try.u -= np.floor(Try.u);      # wraparound to stay within (0,1)
        Try.w -= np.floor(Try.w);      # wraparound to stay within (0,1)
        Try.x = 4.0 * Try.u - 2.0;  # map to x
        Try.z = 2.0 * Try.w;        # map to y
        Try.logL = logLikeHood([[Try.x, Try.z]]);  # trial likelihood value

        # Accept if and only if within hard likelihood constraint
        if Try.logL > logLStar:
            ret.__dict__ = Try.__dict__.copy()
            accept+=1
        else:
            reject+=1

        # Refine step-size to let acceptance ratio converge around 50%
        if( accept > reject ):   step *= np.exp(1.0 / accept);
        if( accept < reject ):   step /= np.exp(1.0 / reject);
    return ret
   




def process_results(results):
    (x,xx) = (0.0, 0.0) # 1st and 2nd moments of x
    (z,zz) = (0.0, 0.0) # 1st and 2nd moments of y
    ni = results['num_iterations']
    samples = results['samples']
    logZ = results['logZ']
    for i in range(ni):
        w = np.exp(samples[i].logWt - logZ); # Proportional weight
        x  += w * samples[i].x;
        xx += w * samples[i].x * samples[i].x;
        z  += w * samples[i].z;
        zz += w * samples[i].z * samples[i].z;
    logZ_sdev = results['logZ_sdev']
    H = results['info_nats']
    H_sdev = results['info_sdev']
    print("# iterates: %i"%ni)
    print("Evidence: ln(Z) = %g +- %g"%(logZ,logZ_sdev))
    print("Information: H  = %g nats = %g bits"%(H,H/np.log(2.0)))
    print("mean(x) = %9.4f, stddev(x) = %9.4f"%(x, np.sqrt(np.abs(xx-x*x))));
    print("mean(z) = %9.4f, stddev(z) = %9.4f"%(z, np.sqrt(np.abs(zz-z*z))));

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
    plt.show()


#----------------- Testing and Running -----------------#
if __name__ == '__main__':
    main()
    