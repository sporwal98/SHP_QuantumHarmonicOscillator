import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class Observe:
    def __init__(self):
        self.sum_x = 0.0
        self.sum_xx = 0.0
        self.xpos = 0.0
        self.xpos2 = 0.0
        self.acintegral = 0.0
        return

    def selfac(self, sitepositions,timeshift, niter,ax = 0):
        result = np.multiply(sitepositions,np.roll(sitepositions,timeshift, axis = ax))
        errs = np.std(result)/niter
        result = np.sum(result,axis = 0)/niter
        return (result,errs)

    def intac(self,ac):
        intac = 0.5*(ac + np.roll(ac,1))
        return np.sum(intac)
        

class Action:
    '''
    Class for action
    '''
    def __init__(self, om = 1):
        self.mass = 1
        self.omega = om
        return

    def harmonic_act_old(self,x):
        #Calculation of Harmonic Oscillator Action
        
        return 0.5*self.mass*self.omega**2*x**2

    def harmonic_act(self, x):
        action = 0.0
        for i in range(len(x)):
            action += (0.5*(x[(i+1)%len(x)]-x[i])**2)+(0.5*self.omega**2*x[i]**2)
        return action

    def del_act(self,xold,xnew,idx):
        del_act = 0.0
        for i in range(idx-1,idx+1):
            oldpart = 0.5*(xold[(i+1)%len(xold)]-xold[i])**2 + 0.5*(self.omega**2)*(xold[i]**2)
            newpart = 0.5*(xnew[(i+1)%len(xnew)]-xnew[i])**2 + 0.5*(self.omega**2)*(xnew[i]**2)
            del_act += newpart-oldpart

        return del_act

class Metropolis:
    '''
    Class for Metropolis time-step
    '''
    def __init__(self, ss = 1e0, act = Action()):
        self.stepsize = ss
        self.action = act
        return

    def step(self,x, idx):
        #One Metropolis step
        acc = False
        tmpx = x.copy()
        #print(x)
        

        #Equally likely to move in positive and negative direction
        step_scale = 2*(np.random.random() - 0.5)
        
        
        x[idx] = x[idx] + step_scale*self.stepsize
        delta_act = self.action.del_act(tmpx,x,idx)
        metro_factor = np.random.random()
        #print('dact = '+str(np.exp(delta_act))+'metro = '+str(metro_factor))
        
        
        if(np.exp(-delta_act)>metro_factor):
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
    


def multisite(output = 'output.txt', ss = 1e0, niterations = 1000, ns = 10, omega = 1,skip = 0, plot = 20):
    #multisite monte carlo
    np.random.seed(42) #random seed for consistent results
    idt = 0 #euclidean time i in {1,...,ntau}
    tmc = 0 #Monte Carlo timesteps

    niter = niterations #number of iterations
    stepsize = ss #increments in position(permitted positions)

    nsites = ns #number of sites(ntau)
    lattice = State(nsites = ns)
    
    act = Action(om = omega) #mass and omega just for generalisation
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
        if(np.mod(i,1000) == 0 ):
            print(i)

        obs.xpos = np.append(obs.xpos, lattice.positions)
        obs.xpos2 = np.append(obs.xpos2, lattice.positions**2)
        ts.append(tmc)
        for j in range(nsites):#for each sweep
            x = lattice.positions.copy()
            xnew, acc = met.step(x,j)
            if(acc):
                naccept[order[j]]+=1
                lattice.positions = xnew
        
        obs.sum_x +=lattice.positions
        obs.sum_xx +=lattice.positions**2
        tmc+=1

    acceptance = np.mean(naccept/niter)
    print('avg acceptance')
    print(acceptance)
    #print(obs.xpos.shape)

    #Setting up the gaussian plot
    mu = obs.sum_x/niter
    ms = obs.sum_xx/niter
    #print('<x>   = '+str(mu))
    #print('<x^2> = '+str(ms))
    var = ms - (mu**2)

    sigma = np.sqrt(var)
    gaussx = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    normal = norm.pdf(gaussx,mu,sigma)

    plt.clf()
    #plt.plot(gaussx,normal,color = 'r',label = 'gaussian')
    plt.plot(np.mean(gaussx,axis = 1),np.mean(normal,axis = 1),color = 'g',label = 'mean gaussian')

    #plt.hist(np.mean(obs.xpos,axis = 1), bins = int(np.sqrt(niter)),label = 'histogram', density = True)
    plt.hist(obs.xpos, bins = int(np.sqrt(niter*nsites)),label = 'histogram', density = True)
    plt.title('Position Histogram: '+ str(niter)+' steps on '+str(nsites)+' sites')
    plt.legend()
    plt.show()

    obs.xpos = np.reshape(obs.xpos, (niter,nsites))
    obs.xpos2 = np.reshape(obs.xpos2, (niter,nsites))
    
    plt.clf()
    plt.plot(np.mean(obs.xpos[skip:],axis = 1),ts[skip:])
    plt.title('Average position of lattice points with time for last '+str(niterations-skip)+' points')
    plt.xlabel('Average position of lattice points')
    plt.ylabel('Time')
    plt.show()
    
    
    
    autocorrs = []
    acerrs = []
    for i in range(niter):
        #autocorrs.append(obs.selfac(obs.xpos,i,niter))
        #autocorrelation of x for each time difference(from 0 to niterations-1)
        
        if(np.mod(i,1000) == 0 ):
            print(i)

        res,err = obs.selfac(obs.xpos,i,niter)
        autocorrs.append(np.mean(res))
        acerrs.append(np.mean(err))
    autocorrs = np.array(autocorrs)
    acerrs = np.array(acerrs)
    #print(autocorrs.shape)

    linefit = np.polyfit(ts[:plot],np.log(autocorrs[:plot]), deg = 1)
    print(linefit)
    line = np.repeat(linefit[1],plot) + np.multiply(np.repeat(linefit[0], plot), np.array(ts[:plot]), dtype = np.dtype(float))

    intautocorr = obs.intac(autocorrs[:plot])
    print('Integrated autocorr = '+str(intautocorr))
    
    poserrors = autocorrs + acerrs
    negerrors = autocorrs - acerrs

    #poserrors*=intautocorr
    #negerrors*=intautocorr
    
    plt.clf()
    #plt.plot(ts[:100],np.log(autocorrs[:100]),label = 'Calculated',marker = '.')
    plt.plot(ts[:plot],autocorrs[:plot],label = 'Autocorrelation',marker = '.')
    #plt.plot(ts[:100],line[:100], c = 'g', label = 'Theoretical')
    plt.plot(ts[:plot],poserrors[:plot],label = 'Positive error',linestyle = 'dashed')
    plt.plot(ts[:plot],negerrors[:plot],label = 'Negative error',linestyle = 'dashed')
    plt.plot(ts[:plot],np.zeros(plot), c = 'r')
    plt.xlabel('Time')
    plt.ylabel('Auto-correlation(for Time difference)')
    plt.legend()
    plt.show()

    plt.clf()
    plt.plot(ts[:plot],np.log(autocorrs[:plot]),label = 'Calculated',marker = '.')
    plt.plot(ts[:plot],np.log(poserrors[:plot]),label = 'Positive error',linestyle = 'dashed')
    plt.plot(ts[:plot],np.log(negerrors[:plot]),label = 'Negative error',linestyle = 'dashed')
    plt.plot(ts[:plot],line[:plot], c = 'c',linestyle = 'dashed', label = 'Theoretical')
    plt.plot(ts[:plot],np.zeros(plot), c = 'r')
    plt.xlabel('Time')
    plt.ylabel('Log Auto-correlation(for Time difference)')
    plt.legend()
    plt.show()


    corrs = []
    errs = []
    for i in range(niter):
        #autocorrs.append(obs.selfac(obs.xpos,i,niter))
        #autocorrelation of x for each time difference(from 0 to niterations-1)
        
        if(np.mod(i,1000) == 0 ):
            print(i)

        res, err = obs.selfac(obs.xpos,i,niter,ax  =1)
        corrs.append(np.mean(res))
        errs.append(np.mean(err))
    corrs = np.array(corrs)
    errs = np.array(errs)

    poserrors = corrs + acerrs
    negerrors = corrs - acerrs

    #poserrors*=intautocorr
    #negerrors*=intautocorr

    plt.clf()
    plt.plot(ts[:plot],corrs[:plot],label = 'Correlation function',marker = '.')
    plt.plot(ts[:plot],poserrors[:plot],label = 'Positive error',linestyle = 'dashed')
    plt.plot(ts[:plot],negerrors[:plot],label = 'Negative error',linestyle = 'dashed')
    plt.plot(ts[:plot],np.zeros(plot), c = 'r')
    plt.xlabel('Time')
    plt.ylabel('Correlation function(for Time difference)')
    plt.legend()
    plt.show()

    plt.clf()
    plt.plot(ts[:plot],np.log(corrs[:plot]),label = 'Log Correlation function',marker = '.')
    plt.plot(ts[:plot],np.log(poserrors[:plot]),label = 'Positive error',linestyle = 'dashed')
    plt.plot(ts[:plot],np.log(negerrors[:plot]),label = 'Negative error',linestyle = 'dashed')
    plt.plot(ts[:plot],np.zeros(plot), c = 'r')
    plt.xlabel('Time')
    plt.ylabel('Log Correlation function(for Time difference)')
    plt.legend()
    plt.show()
    #print(autocorrs.shape)

    
    
    
    
    return


    
