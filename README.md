# SHP_QuantumHarmonicOscillator
UPDATE: New structure of code:
1. 'main' method for simulation
2. 'Action' class to implement action of various systems.
3. 'Observe' class to implement observables
4. 'Metropolis' class to implement Metropolis time-step


This project simulates the functioning of a 1-D Quantum Harmonic Oscillator, using a Markov Chain Monte Carlo Method.
The current structure of the code is such:
1. 'main' method for main simulation and recording observables(WIP)
2. 'harmonic_osc' method for iteration(WIP)
3. 'calc_action' method for calculation of action(WIP, need to check if correct)
4. 'calc_ham' method for calculation of hamiltonian
5. 'calc_delh' method for calculation of derivative of hamiltonian wrt x
