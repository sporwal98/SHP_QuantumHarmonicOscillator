import numpy as np
import matplotlib.pyplot as plt

class Observe:
    def __init__(self):
        return

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
        return 0.5*self.mass*self.omega* x**2
    

class Metropolis:
    '''
    Class for Metropolis time-step
    '''
    def __init__(self, ls = 1e0, act = Action()):
        self.latticespace = ls
        self.action = act
        return

    def step(self,x):
        #One Metropolis step
        acc = False
        tmpx = x

        #Equally likely to move in positive and negative direction
        sign = np.random.random()
        if(sign>=0.5):
            sign = +1
        else:
            sign = -1
        
        
        x = x + sign*self.latticespace
        delta_act = -(self.action.harmonic_act(x)-self.action.harmonic_act(tmpx))
        metro_factor = np.random.random()
        
        
        if(np.exp(delta_act)>metro_factor):
            acc = True
        else:
            x = tmpx
        return x, acc

def main(output = 'output.txt', ls = 1e0, niterations = 1000, skip = 0):
    '''
    Main simulation code
    '''
    np.random.seed(42) #random seed for consistent results
    idt = 0 #euclidean time i in {1,...,ntau}
    tmc = 0 #Monte Carlo timesteps

    niter = niterations #number of iterations
    latticespace = ls #increments in position(permitted positions)
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
    #sum_x = 0.0 #sum of x, used for <x>
    #sum_xx = 0.0 #sum of x^2, used for <x^2>

    act = Action(m = 1, om = 1)
    met = Metropolis(ls = latticespace, act = act)

    for iter in range(niter):
    #For each monte carlo timestep
        x, acc = met.step(x)
        tmc += 1

        if(acc):
            naccept += 1

        xs.append(x)
        ts.append(tmc)

        #sum_x += x
        #sum_xx += x^2
        #outfile.write('DATA')

    #outfile.close()
    
    plt.clf()
    plt.plot(xs[skip:],ts[skip:])
    plt.xlabel('Position x')
    plt.ylabel('Monte Carlo Timestep')
    plt.title('Acceptance = '+str(naccept/niter))
    plt.show()
    
    return 0
