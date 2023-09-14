import numpy as np
from qiskit import QuantumCircuit
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import random_unitary as U
from scipy.spatial.distance import hamming
from qiskit import Aer, execute # Use AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
simulator = Aer.get_backend('qasm_simulator')

# Randomized method to find 2-order Rényi entropy. Not purely theoretical, but nshots are made.


class Randomized_Renyi_2():
    
    def __init__(self) -> None:
        self.entropy_value = None
        self.entropy_error = None
        self.number_measurements = None
        self.number_A = None
        self.getU_matrix = None
        self.freq_list = None
        self.original1 = None
        self.p3 = None

    def Random_Circuit(self, circuit: QuantumCircuit, N_A: list):
        c = circuit
    
        #Random unitary gates are applied.     
        for i in N_A:
            U_gate = U(2)
            gate = UnitaryGate(U_gate)
            c.append(gate, [i])
        c.measure(N_A,range(len(N_A)))

        return c

    def X_fun(self, freq, N_A, N_M):
        '''
        This functions estimates the purtiy of subsystem A, tr(rho_A**2), with a fixed set of local random unitaries, for N_A a list containing 
        the qubits that form part of such subspace.
        Parameters:
            circuit: circuit that represents the initial state we want to estimate the entropy of.
            N_A: list of qubits that constitue subspace A. 
            N_M: number of measurement per each unitary ensamble.
        '''
        
        #Formula is applied, summing over all probabilities twice and computing the hamming distance.
        x_list = []
        x_error = []
        for i in freq:
            j_iter = 0
            j_error = 0
            for j in freq:
                count = 0
                count_error = 0
                d = hamming(list(str(i)), list(str(j)))*len(N_A)
                freq[i] = freq[i]*N_M
                freq[j] = freq[j]*N_M
                if i == j:
                    count = (-2)**(-d)*(((freq[i]/N_M)*(freq[i] - 1))/(N_M - 1))
                    count_error = (-2)**(-d)*(2*freq[i] - 1)/(N_M - 1)*np.sqrt((freq[i]/N_M)*(1-(freq[i]/N_M))/N_M)
                else:
                    count = (-2)**(-d)*freq[i]*freq[j]/(N_M**2)
                    count_error = (-2)**(-d)*np.sqrt(((freq[j]/N_M)*np.sqrt((freq[i]/N_M)*(1-(freq[i]/N_M))/N_M))**2 + ((freq[i]/N_M)*np.sqrt((freq[j]/N_M)*(1-(freq[j]/N_M))/N_M))**2)
                    
                j_iter += count
                j_error += count_error**2
            x_list.append(j_iter)
            x_error.append(np.sqrt(j_error))

        X = 2**len(N_A)*np.sum(x_list)
        x_error = np.array(x_error)
        X_error = 2**len(N_A)*np.sqrt(np.sum(x_error**2))
        
        return X , X_error, freq
    
    def Local_simulation(self, circuit: QuantumCircuit, N_A: list, N_U: int, N_M: int):
        '''
        It computes the 2-Réyni entropy of the circuit following the local randomized unitary protocol.
        Parameters:
            circuit: initial circuit where the funcion is applied. It must represent the state we want to estimate the entropy of.
            N_A: list of qubits' position that constitue subspace A.
            N_U: number of fixed local unitary ensambles we are performing.
            N_M: number of measurements for each unitary ensamble.
        '''
        super_list = []
        super_error = []
        super_freq = []
        qc_list = []

        for _ in range(N_U):
            c_copy = circuit.copy()
            random_circuit = self.Random_Circuit(c_copy, N_A)
            qc_list.append(random_circuit)

        job = execute(qc_list, simulator, shots = N_M)
        result = job.result()
        counts = result.get_counts()
        
        for i in range(N_U):
            X, X_error, freq = self.X_fun(counts[i], N_A, N_M)
            super_list.append(X)
            super_error.append(X_error)
            super_freq.append(freq)
        X_mean = np.mean(super_list)
        super_error = np.array(super_error)
        X_std = 1/N_U*np.sqrt(np.sum(super_error**2))
        S_2 = - np.log2(X_mean)
        S_2_error = X_std/(X_mean*np.log(2))


        
        self.entropy_value = S_2
        self.entropy_error = S_2_error
        self.freq_list = super_freq

    def Local_hardware(self, circuit: QuantumCircuit, N_A: list, N_U: int, N_M: int, backend:str, token:str):
        '''
        It computes the 2-Réyni entropy of the circuit following the local randomized unitary protocol.
        Parameters:
            circuit: initial circuit where the funcion is applied. It must represent the state we want to estimate the entropy of.
            N_A: list of qubits' position that constitue subspace A.
            N_U: number of fixed local unitary ensambles we are performing.
            N_M: number of measurements for each unitary ensamble.
        '''
        super_list = []
        super_error = []
        super_freq = []
        qc_list = []
        # Get the API token in https://quantum-computing.ibm.com/account
        service = QiskitRuntimeService(channel="ibm_quantum", token=token)
        backend = service.backend(backend)
        sampler = Sampler(session=backend)

        for _ in range(N_U):
            c_copy = circuit.copy()
            random_circuit = self.Random_Circuit(c_copy, N_A)
            qc_list.append(random_circuit)

        # Run the circuit and get result distribution
        job = sampler.run(qc_list, shots = N_M)
        quasi_dist = job.result().quasi_dists
        binary = quasi_dist.binary_probabilities()
        
        for i in range(N_U):
            X, X_error, freq = self.X_fun(binary[i], N_A, N_M)
            super_list.append(X)
            super_error.append(X_error)
            super_freq.append(freq)
        X_mean = np.mean(super_list)
        super_error = np.array(super_error)
        X_std = 1/N_U*np.sqrt(np.sum(super_error**2))
        S_2 = - np.log2(X_mean)
        S_2_error = X_std/(X_mean*np.log(2))


        
        self.entropy_value = S_2
        self.entropy_error = S_2_error
        self.freq_list = super_freq