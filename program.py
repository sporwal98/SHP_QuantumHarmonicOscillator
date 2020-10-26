import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class Observe:
    def __init__(self):
        self.sum_x = 0.0
        self.sum_xx = 0.0
        self.xpos = 0.0
        self.xpos2 = 0.0
        return

    def selfac(self, sitepositions,timeshift, niter):
        result = np.multiply(sitepositions,np.roll(sitepositions,timeshift*niter))        
        result = np.sum(result,axis = 0)/niter
        return result

class Action:
    '''
    Class for action
    '''
    def __init__(self, m = 1, om = 1):
        self.mass = m
        self.omega = om
        return

    def harmonic_act(self,x):
        #Calculation of Harmonic Oscillator Action
        
        return 0.5*self.mass*self.omega**2*x**2 #+ self.diff_actionwrtx(x-np.roll(x,1)) 
        

    def diff_actionwrtx(self,x):
        #Differential of action wrt x
        return self.mass*self.omega*x
    
    
        

class Metropolis:
    '''
    Class for Metropolis time-step
    '''
    def __init__(self, ss = 1e0, act = Action()):
        self.stepsize = ss
        self.action = act
        return

    def step(self,x):
        #One Metropolis step
        acc = False
        tmpx = x

        #Equally likely to move in positive and negative direction
        step_scale = 2*(np.random.random() - 0.5)
        
        
        x = x + step_scale*self.stepsize
        delta_act = -(self.action.harmonic_act(x)-self.action.harmonic_act(tmpx))
        metro_factor = np.random.random()
        
        
        if(np.exp(delta_act)>metro_factor):
            acc = True
        else:
            x = tmpx
        return x, acc

class State:
    '''
    Class for State of multisite QHO
    '''
    def __init__(self,nsites, omega = 1, mass = 1):
        self.positions = np.zeros(nsites)
        self.omega = omega
        self.mass = mass
    

def singlesite(output = 'output.txt', ss = 1e0, niterations = 1000, skip = 0):
    '''
    Main simulation code
    '''
    np.random.seed(42) #random seed for consistent results
    idt = 0 #euclidean time i in {1,...,ntau}
    tmc = 0 #Monte Carlo timesteps

    niter = niterations #number of iterations
    stepsize = ss #increments in position(permitted positions)
    x = 0.0 #initial position
    tmc = 0 #monte carlo time-step
        
    xs = [x] #list of x positions
    ts = [tmc] #list of time-steps
    
    '''
    #open output file
    outfile = open('output.txt','w')
    outfile.write('HEADERS')
    '''
    
    naccept = 0 #counter for acceptances
    sum_x = 0.0 #sum of x, used for <x>
    sum_xx = 0.0 #sum of x^2, used for <x^2>

    act = Action(m = 1, om = 1) #mass and omega just for generalisation
    met = Metropolis(ss = stepsize, act = act) 

    energies = [act.harmonic_act(x)]
    
    for iter in range(niter):
    #For each monte carlo timestep
        x, acc = met.step(x)
        tmc += 1
        

        if(acc):
            naccept += 1

        xs.append(x)
        ts.append(tmc)
        energies.append(act.harmonic_act(x))
        
        sum_x += x
        sum_xx += x**2
        #outfile.write('DATA')

    #outfile.close()

    '''   
    plt.clf()
    plt.plot(xs[skip:],ts[skip:])
    plt.xlabel('Position x')
    plt.ylabel('Monte Carlo Timestep')
    plt.title('Acceptance = '+str(naccept/niter))
    plt.show()

    plt.clf()
    plt.plot(ts[skip:], energies[skip:])
    plt.xlabel('Monte Carlo Timestep')
    plt.ylabel('Action')
    plt.title('Energies')
    plt.show()
    '''

    mu = sum_x/niter
    rms = sum_xx/niter
    var = rms - (mu**2)

    sigma = np.sqrt(var)
    gaussx = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    normal = np.exp(-gaussx**2/2*var)/(sigma*np.sqrt(2*np.pi))
    
    
    plt.clf()
    plt.plot(gaussx, normal,color = 'r',label = 'gaussian')
    plt.hist(xs, bins = int(np.sqrt(niter)),label = 'histogram', density = True)
    plt.title('Position Histogram and Gaussian '+ str(niter))
    plt.legend()
    plt.show()
    
    print('<x> = ' + str(mu))
    print('<x^2> = ' + str(rms))
    
    


    
    return 0

def multisite(output = 'output.txt', ss = 1e0, niterations = 1000, ns = 10, skip = 0):
    #multisite monte carlo
    np.random.seed(42) #random seed for consistent results
    idt = 0 #euclidean time i in {1,...,ntau}
    tmc = 0 #Monte Carlo timesteps

    niter = niterations #number of iterations
    stepsize = ss #increments in position(permitted positions)

    nsites = ns #number of sites(ntau)
    lattice = State(nsites = ns)
    
    act = Action(m = 1, om = 1) #mass and omega just for generalisation
    met = Metropolis(ss = stepsize, act = act) #Metropolis object
    obs = Observe() # Object for observables.
    
    naccept = np.zeros(nsites) #counter for acceptances
    obs.sum_x = np.zeros(nsites) #sum of x, used for <x>
    obs.sum_xx = np.zeros(nsites) #sum of x^2, used for <x^2>
    obs.xpos = np.array([])#store x positions for each time step
    obs.xpos2 = np.array([])#store x^2 for each time step
    ts = []

    for i in range(niter):#number of sweeps
        order = np.random.permutation(nsites)

        obs.xpos = np.append(obs.xpos, lattice.positions)
        obs.xpos2 = np.append(obs.xpos2, lattice.positions**2)
        ts.append(tmc)
        for j in range(nsites):#for each sweep
            x = lattice.positions[order[j]]
            xnew, acc = met.step(x)
            if(acc):
                naccept[order[j]]+=1
                lattice.positions[order[j]] = xnew
        
        obs.sum_x +=lattice.positions
        obs.sum_xx +=lattice.positions**2
        tmc+=1

    acceptance = np.mean(naccept/niter)
    print('avg acceptance')
    print(acceptance)
    #print(obs.xpos.shape)
    
    mu = obs.sum_x/niter
    rms = obs.sum_xx/niter
    var = rms - (mu**2)

    sigma = np.sqrt(var)
    gaussx = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    normal = np.exp(-gaussx**2/2*var)/(sigma*np.sqrt(2*np.pi))

    plt.clf()
    #plt.plot(gaussx,np.mean(normal,axis = 1),color = 'r',label = 'gaussian')
    plt.plot(np.mean(gaussx,axis = 1),np.mean(normal,axis = 1),color = 'g',label = 'mean gaussian')
    plt.hist(obs.xpos, bins = int(np.sqrt(niter*nsites)),label = 'histogram', density = True)
    plt.title('Position Histogram: '+ str(niter)+' steps on '+str(nsites)+' sites')
    plt.legend()
    plt.show()

    obs.xpos = np.reshape(obs.xpos, (niter,nsites))
    obs.xpos2 = np.reshape(obs.xpos2, (niter,nsites))

    autocorrs = []
    for i in range(niter):
        #autocorrs.append(obs.selfac(obs.xpos,i,niter))
        #autocorrelation of x for each time difference(from 0 to niterations-1)
        '''
        if(np.mod(i,10) == 0 ):
            print(i)
        '''
        autocorrs.append(np.mean(obs.selfac(obs.xpos,i,niter)))
    autocorrs = np.array(autocorrs)
    print(autocorrs.shape)
    plt.clf()
    plt.plot(ts[skip:],autocorrs[skip:])
    plt.xlabel('Time')
    plt.ylabel('Autocorrelation(for Time difference)')
    plt.show()

    plt.clf()

    linefit = np.polyfit(np.mean(obs.xpos,axis = 1),autocorrs, deg = 1)
    print(linefit)

    return 0


    
