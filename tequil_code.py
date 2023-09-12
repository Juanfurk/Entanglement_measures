import numpy as np
import tequila as tq
from qibo.quantum_info import random_unitary as U
from scipy.spatial.distance import hamming
from qibo.models import Circuit
from qiskit.quantum_info import OneQubitEulerDecomposer as OQED 
from collections import Counter
from tequila.circuit import QCircuit
import itertools
import copy

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

    def int_to_binary_string(self, number: int, nqubits:int):
        '''
        Given a number, it returns its associated bitstring representation, with total sizew of nqubits.
        Parameters:
            number: any number, as log as it tis int.
            nqubits: total number of qubits.
        '''
        if number < 0:
            raise ValueError("Input must be a non-negative integer.")
        elif number == 0:
            zero_str = '0'
            string = ''
            for _ in range(nqubits - 1):
                string += zero_str 

            return string + "0"  # Special case for 0

        binary_string = bin(number)
        zero_str = '0'
        total = binary_string[2:]
        while len(total) < nqubits:
                total = zero_str + total 
        return total

    def dict_freq(self, wfn, nqubits:int):
        '''
        This function allows to take the results given by wfn.items() and convert them into another dictionary
        with bitstrings (str) associated to the counts (int).
        Parameters:
            wfn: tequila wavefunction (comes from the simulation object)
            nqubits: number of qubits on subspace A (qubits we want to measure).
        '''
        items = wfn.items()     #gives a dictionary of counts associated to a number (which is the bitstring count)
        bit_strings = []
        counts = []
        for num, freq in items:
            num = int(num)
            freq = int(freq)
            bit = self.int_to_binary_string(num, nqubits)   #turns the number given by wfn.item() into a bitsring (str)
            bit_strings.append(bit)
            counts.append(freq)

        dict_counts = {key: value for key, value in zip(bit_strings, counts)}
        return Counter(dict_counts)

    def X_fun(self, circuit: QCircuit, N_A: list, N_M: int, backend: str):
        '''
        This functions estimates the purtiy of subsystem A, tr(rho_A**2), with a fixed set of local random unitaries, for N_A a list containing 
        the qubits that form part of such subspace.
        Parameters:
            circuit: circuit that represents the initial state we want to estimate the entropy of.
            N_A: list of qubits that constitue subspace A. 
            N_M: number of measurement per each unitary ensamble.
            backend: backend used during tequila's simulation.
        '''
        c = circuit
        gate_list = []
        #Random unitary gates are applied.     
        for i in N_A:
            U_gate = U(2, 'haar')
            angles = OQED(basis = 'U3').angles(U_gate)
            gate_list.append(U_gate)
            c += tq.gates.u3(theta=angles[0], phi = angles[1], lambd= angles[2], target = i)
        wfn = tq.simulate(c, samples=N_M, backend = backend , read_out_qubits = N_A)
        freq = self.dict_freq(wfn, len(N_A)) #importante para trabajar con las frecuencias y bitstrings una vez medido


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
        gate_list = np.array(gate_list)

        return X , X_error, gate_list, freq

    def Local(self, circuit: QCircuit, N_A: list, N_U: int, N_M: int, backend: str):
        '''
        It computes the 2-Réyni entropy of the circuit following the local randomized unitary protocol.
        Parameters:
            circuit: initial circuit where the funcion is applied. It must represent the state we want to estimate the entropy of.
            N_A: list of qubits' position that constitue subspace A.
            N_U: number of fixed local unitary ensambles we are performing.
            N_M: number of measurements for each unitary ensamble.
            backend: backend used during tequila's simulation.
        '''
        if backend == None: 
            backend = 'qulacs'
        super_list = []
        super_error = []
        super_gate_list = []
        super_freq = []
        for _ in range(N_U):
            c_copy = copy.deepcopy(circuit)
            X, X_error, U_gates, freq = self.X_fun(c_copy, N_A, N_M, backend)
            super_list.append(X)
            super_error.append(X_error)
            super_gate_list.append(U_gates)
            super_freq.append(freq)
        X_mean = np.mean(super_list)
        super_error = np.array(super_error)
        X_std = 1/N_U*np.sqrt(np.sum(super_error**2))
        S_2 = - np.log2(X_mean)
        S_2_error = X_std/(X_mean*np.log(2))

        super_gate_list = np.array(super_gate_list)
        super_gate_list = np.reshape(super_gate_list, (N_U, len(N_A), 4))

        
        self.entropy_value = S_2
        self.entropy_error = S_2_error
        self.getU_matrix = super_gate_list
        self.freq_list = super_freq
        

    def Global_Unitary(self, circuit: QCircuit, N_A:list):       
        '''
        Given a circuit, this function adds, consecutively, Hadamard gates on all qubits and the so-called 
        Random Diagonal Circuit (RDI), in an iteratively way O(N^(1/2)). It will work as our Global Random Unitary, 
        as iterate H + RCD generate an epsilon-approximate t-design.
        Parameters:
            circuit: Qcircuit where we add the Global Unitary gate (gobal measurements). 
            N_A: list containing the qubits of subspace A.
        ''' 
        c = circuit
        nqubits = len(N_A)
        for _ in range(int(2*np.sqrt(nqubits)) + 1):
            for i in N_A:
                c += tq.gates.H(target=i) 
                U_gate = U(2, 'haar')
                angles = OQED(basis = 'U3').angles(U_gate)
                c += tq.gates.u3(theta=angles[0], phi = angles[1], lambd= angles[2], target = i)

            for j in N_A:
                for i in N_A:
                    if j >= i:
                        pass
                    else:
                        U_gate = U(2, 'haar')
                        angles = OQED(basis = 'U3').angles(U_gate)
                        c += tq.gates.u3(theta=angles[0], phi = angles[1], lambd= angles[2], target = i, control= j)

        return c

    def X_fun_global(self, circuit: Circuit, N_A: list, N_M: int, backend:str):
        '''
        This functions estimates the purtiy of subsystem A tr(rho_A**2) with a fixed set of global random unitaries, for N_A a list containing the qubits that form
        part of such subspace.
        Parameters:
            circuit: circuit that represents the initial state we want to estimate the entropy of.
            N_A: list of qubits that constitue subspace A. 
            N_M: number of measurement per each unitary ensamble.
            backend: backend used during tequila's simulation.
        '''
        c = self.Global_Unitary(circuit, N_A)

        wfn = tq.simulate(c, samples=N_M, backend = backend , read_out_qubits = N_A)
        freq = self.dict_freq(wfn, len(N_A)) #importante para trabajar con las frecuencias una vez medido

        #Formula is applied, summing over all probabilities twice and computing the hamming distance.
        count = 0
        count_error = 0
        for i in freq:
            count += ((freq[i]/N_M)*(freq[i] - 1))/(N_M - 1)
            count_error += ((2*freq[i] - 1)/(N_M - 1)*np.sqrt((freq[i]/N_M)*(1-(freq[i]/N_M))/N_M))**2

        X = (2**len(N_A) + 1)*count - 1
        X_error = (2**len(N_A) + 1)*np.sqrt(count_error)
        
        return X, X_error, freq

    def Global(self, circuit: Circuit, N_A: list, N_U: int, N_M: int, backend:str):
        '''
        It computes the 2-Réyni entropy of the circuit following the global randomized unitary protocol.
        Parameters:
            circuit: initial circuit where the function is applied. It must represent the state we want to estimate the entropy of.
            N_A: list of qubits that constitue subspace A.
            N_U: number of fixed local unitary ensambles we are performing.
            N_M: number of measurements for each unitary ensamble.
            backend: backend used during tequila's simulation.
        '''
        super_list = []
        #super_gate_list = []
        super_freq = []
        super_error = []
        self.number_measurements = N_M
        self.number_A = N_A
        for _ in range(N_U):
            c_copy = copy.deepcopy(circuit)
            X, X_error, freq = self.X_fun_global(c_copy, N_A, N_M, backend)
            super_list.append(X)
            super_error.append(X_error)
            #super_gate_list.append(U_gates)
            super_freq.append(freq)
        X_mean = np.mean(super_list)
        super_error = np.array(super_error)
        X_std = 1/N_U*np.sqrt(np.sum(super_error**2))
        S_2 = - np.log2(X_mean)
        S_2_error = X_std/(X_mean*np.log(2))

        #super_gate_list = np.array(super_gate_list)
        #super_gate_list = np.reshape(super_gate_list, (N_U, len(N_A), 4))

        
        self.entropy_value = S_2
        self.entropy_error = S_2_error
        #self.getU_matrix = super_gate_list
        self.freq_list = super_freq
    

    def Clifford(self, circuit: QCircuit, N_A: list, N_M: int, backend:str):
        '''
        It computes the 2-Renyi entropy using the average over the Clifford group.
        Parameters:
            circuit: initial circuit where the funcion is applied. It must represent a state where we want to estimate the entropy.
            N_A: number of qubits in subspace A. Belongs only to the first part of the circuit.
            N_M: number of measurements.
            backend: backend used during tequila's simulation.
        '''

        H = 1/np.sqrt(2)*np.array([[1,1],[1,-1]], dtype=complex)
        S = np.array([[1,0],[0,1j]])

        single_qubit_cliffords = [
            '',
            'H', 'S',
            'HS', 'SH', 'SS',
            'HSH', 'HSS', 'SHS', 'SSH', 'SSS',
            'HSHS', 'HSSH', 'HSSS', 'SHSS', 'SSHS',
            'HSHSS', 'HSSHS', 'SHSSH', 'SHSSS', 'SSHSS',
            'HSHSSH', 'HSHSSS', 'HSSHSS'
        ]

        matrix_products = []

        for string in single_qubit_cliffords:
            product = np.eye(2)
            for symbol in string:
                if symbol == 'H':
                    product = product @ H

                if symbol == 'S':
                    product = product @ S

                else: 
                    pass
            matrix_products.append(product)

        permutations_matrices = list(itertools.product(matrix_products, repeat=len(N_A)))

        super_list = []
        super_error = []

        for perm in permutations_matrices:
            c_copy = copy.deepcopy(circuit)

            for i, j in enumerate(N_A):
                Clifford_gate = perm[i]
                angles = OQED(basis = 'U3').angles(Clifford_gate)
                c_copy += tq.gates.u3(theta=angles[0], phi = angles[1], lambd= angles[2], target = j)


            wfn = tq.simulate(c_copy, samples=N_M, backend = backend , read_out_qubits = N_A)
            freq = self.dict_freq(wfn, len(N_A)) #importante para trabajar con las frecuencias una vez medido
            #Formula is applied, summing over all probabilities twice and computing the hamming distance.
            x_list = []
            x_error = []
            for i in freq:
                j_iter = 0
                j_error = 0
                for j in freq:
                    count = 0
                    d = hamming(list(str(i)), list(str(j)))*len(N_A)
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
            super_list.append(X)
            super_error.append(X_error)
                               
    
        X_mean = np.mean(super_list)
        super_error = np.array(super_error)
        X_std = 1/len(permutations_matrices)*np.sqrt(np.sum(super_error**2))

        S_2 = - np.log2(X_mean)
        S_2_error = X_std/(X_mean*np.log(2))

        
        self.entropy_value = S_2
        self.entropy_error = S_2_error


    def entropy(self):
        '''
        It returns the estimated entropy of the especified randomized protocol.
        '''
        return self.entropy_value
    
    def error(self):
        '''
        It returns the estimated standard error of the entropy estimation for the randomized protocol.
        '''
        return self.entropy_error
    
    def get_U(self, iter, num):
        '''
        It returns the set of random unitaries of the local randomized protocol.
        '''
        return np.array(self.getU_matrix[iter, num, :]).reshape(2,2)
    
    def frequencies(self, iter = None):
        '''
        It returns the outcomes frequencies of the randomized protocol.
        Parameters:
            iter: number of iteration of the ensemble of random unitaries apply. If not specified,
                the functions just returns all the frequencies of all iterarions N_U.
        '''
        if iter is None:
            return self.freq_list
        else:
            return self.freq_list[iter]
        
