import numpy as np
import matplotlib.pyplot as plt

ntau = 40 #elements in time lattice
idt = 0 #euclidean time i in {1,...,ntau}
tmc = 0 #Monte Carlo time
n_paths= 0 #Number of paths in ensemble
nbins = 0 # Number of bins in Jackknife

niter = 10000#number of iterations
dtau = 1e0#lattice spacing

xs = [] #list of x positions


def main():
#PLACEHOLDER
#Main simulation code
    x = 0 #effective(x/dtau)
    backup_x = 0 #temporary variable for cases where change is rejected
    ham_in = 0 #initial hamiltonian(just declaring)
    ham_fin = 0 #final hamiltonian(just declaring)
    metropolis = 0 #metropolis factor(just declaring

    #parameters needed
    m = 1 #mass(effective, m*dtau)
    omega = 1 # = sqrt(k/m) where k is force constant(effective, omega*dtau
    
    
    

    #initial config
    x = 0.0
    xs.append(x)
    
    #open output file
    outfile = open('output.txt','w')
    outfile.write('HEADERS')

    naccept = 0 #counter for acceptances
    sum_xx = 0.0 #sum of x^2, used for <x^2>

    for iter in range(0, niter):
    #for each iteration
        backup_x = x
        x, ham_in, ham_fin = harmonic_osc(x, ham_in,ham_fin, m, omega)
        metropolis = np.random.random()

        if(np.exp(ham_in - ham_fin) > metropolis):
        #accept
            naccept = naccept + 1
        else:
        #reject
            x = backup_x

        sum_xx = sum_xx + x*x

        xs.append(x)
        
        outfile.write('DATA')

    outfile.close()
    
            
        

    
    
    return 0

def harmonic_osc(x, ham_in, ham_fin, m,omega):
#PLACEHOLDER
#CHANGE IN SYSTEM
    para = 'PLACEHOLDER'

    rand1 = np.random.random()
    rand2 = np.random.random()

    #p value not finalised
    p = rand1
    
    ham_in = calc_ham(m, omega, x, p)

    #ITERATION OF X NEEDS TO CHANGE
    x = x + p*0.5*dtau
    
    for step in range(1, ntau):
    #ITERATIONS NEED CITATION
        delh = calc_delh(m, omega, x)
        p = p - delh*dtau
        x = x + p*0.5*dtau

    delh = calc_delh(para)
    p = p - delh 
    x = x + p*0.5*dtau

    ham_fin = calc_ham(m, omega, x, p)

    return x, ham_in, ham_fin    

    

def calc_action(m, omega, x):
#PLACEHOLDER
#Calculation of action. Change must be reflected in calc_delh
    act = m*omega*omega*x*x/2
    return act

def calc_ham(m, omega, x, p):
#PLACEHOLDER
#Calculation of action
    ham = calc_action(m,omega,x)
    ham = ham + p*p/(2*m)

    return ham

def calc_delh(m,omega,x):
#PLACEHOLDER
#Calculation of derivative of Hamiltonian wrt x
    delh = m*omega*omega*x
    return delh
