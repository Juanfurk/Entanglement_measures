import numpy as np
from qibo import  gates
from qibo.quantum_info import random_unitary as U
from scipy.spatial.distance import hamming
from qibo.models import Circuit

# Randomized method to find 2-order Rényi entropy. Not purely theoretical, but nshots are made.


class Randomized_Renyi_2():
    
    def __init__(self) -> None:
        self.entropy_value = None
        self.entropy_error = None
        self.getU_matrix = None
        self.freq_list = None
        self.original1 = None

    def X_fun(self, circuit: Circuit, N_A: list, N_M: int):
        '''
        This functions estimates the purtiy of subsystem A, tr(rho_A**2), with a fixed set of local random unitaries, for N_A a list containing 
        the qubits that form part of such subspace.
        Parameters:
            circuit: circuit that represents the initial state we want to estimate the entropy of.
            N_A: list of qubits that constitue subspace A. 
            N_M: number of measurement per each unitary ensamble.
        '''
        c = circuit
        gate_list = []
        #Random unitary gates are applied.     
        for i in N_A:
            U_gate = U(2, 'haar')
            gate_list.append(U_gate)
            c.add(gates.Unitary(U_gate, i))
            c.add(gates.M(i))
        result = c(nshots=N_M)
        freq = result.frequencies()
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

    def Local(self, circuit: Circuit, N_A: list, N_U: int, N_M: int):
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
        super_gate_list = []
        super_freq = []
        for _ in range(N_U):
            c_copy = circuit.copy()
            X, X_error, U_gates, freq = self.X_fun(c_copy, N_A, N_M)
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
    
    def Global_Unitary(self, nqubits):       
        '''
        It computes, consecutively, Hadamard gates on all qubits and the so-called Random Diagonal Circuit (RDI), in an iteratively way O(N^(1/2)). 
        It will work as our Global Random Unitary, as iterate H + RCD generate an epsilon-approximate t-design.
        Parameters:
            nqubits: number of qubits of subspace A.
        ''' 
        c = Circuit(nqubits)
        for _ in range(int(2*np.sqrt(nqubits)) + 1):
            for i in range(nqubits):
                c.add(gates.H(i))
                U_gate = U(2, 'haar')
                c.add(gates.Unitary(U_gate,i))

            for j in range(1, nqubits):
                for i in range(j,nqubits):
                    U_gate = U(2, 'haar')
                    c.add(gates.Unitary(U_gate, i).controlled_by(j-1))

        return c

    def X_fun_global(self, circuit: Circuit, N_A: list, N_M: int):
        '''
        This functions estimates the purtiy of subsystem A tr(rho_A**2) with a fixed set of global random unitaries, for N_A a list containing the qubits that form
        part of such subspace.
        Parameters:
            circuit: circuit that represents the initial state we want to estimate the entropy of.
            N_A: list of qubits that constitue subspace A. 
            N_M: number of measurement per each unitary ensamble.
        '''
        c = circuit
        #Random unitary gates are applied.   
        U_global_circ = self.Global_Unitary(len(N_A))
        c.add(U_global_circ.on_qubits(*N_A))

        for i in N_A:
            c.add(gates.M(i))
            
        result = c(nshots=N_M)
        freq = result.frequencies()
        #Formula is applied, summing over all probabilities twice and computing the hamming distance.
        count = 0
        count_error = 0
        for i in freq:
            count += ((freq[i]/N_M)*(freq[i] - 1))/(N_M - 1)
            count_error += ((2*freq[i] - 1)/(N_M - 1)*np.sqrt((freq[i]/N_M)*(1-(freq[i]/N_M))/N_M))**2

        X = (2**len(N_A) + 1)*count - 1
        X_error = (2**len(N_A) + 1)*np.sqrt(count_error)
        
        return X, X_error, freq

    def Global(self, circuit: Circuit, N_A: list, N_U: int, N_M: int):
        '''
        It computes the 2-Réyni entropy of the circuit following the global randomized unitary protocol.
        Parameters:
            circuit: initial circuit where the function is applied. It must represent the state we want to estimate the entropy of.
            N_A: list of qubits that constitue subspace A.
            N_U: number of fixed local unitary ensambles we are performing.
            N_M: number of measurements for each unitary ensamble.
        '''
        super_list = []
        #super_gate_list = []
        super_freq = []
        super_error = []
        for _ in range(N_U):
            c_copy = circuit.copy()
            X, X_error, freq = self.X_fun_global(c_copy, N_A, N_M)
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
        

#Protocol to estimate the n-order Rényi entropy. There is an exact method (which requires too much computational power) and 
#the measurement method, where N_M measurements are performed on the auxiliary qubit.

class Swap_Renyi_n():
    def __init__(self) -> None:
        self.entropy_value = None
        self.entropy_error = None
        self.freq_list = None

    def Exact(self, n : int, d : int, N_A : list, circuit : Circuit):
        '''
        Given a circuit of d qubits that represents a system with two subsystems A and B, 
        it returns the exact n-Renyi entropy of the system, without any measurement. 
        
        Parameters:
            n: order of the entropy wished to mesure.
            state_size: list of qubits that form the initial state. Must be size 2; first element indicates number of qubits for subspace A 
            and second elements the number of qubits for subspace B.
            circuit: initial system that represents the desired state.
        '''

        circuit1 = circuit
        circuit2 = Circuit(n*d + 1)
        for l in range(n):
            circuit2.add(circuit1.on_qubits(*range(l*d + 1, (l+1)*d + 1)))
        c = circuit2
        c.add(gates.H(0))
        for l in range(n-1):
            for a in N_A:
                c.add(gates.SWAP(l*d + a + 1, (l+1)*d + a + 1).controlled_by(0))

        c.add(gates.H(0))

        c.add(gates.M(0))

        result = c() #we can always initialize the state in any density matrix by seting result = c(density_matrix), and circuit = Circuit(density_matrix = True)
        psi = np.array(result)
        psi = psi.reshape(-1,1)
        prob_0 = 0
        prob_1 = 0
        dim = int(psi.size)
        for i in range(0,int(dim/2)):
            prob_0 += np.absolute(psi[i])**2
        for j in range(int(dim/2),dim):
            prob_1 += np.absolute(psi[j])**2
        R_n = prob_0 - prob_1
        S_n = -np.log2(R_n)
        self.entropy_value = S_n


    def Measurement(self, n : int, d : int, N_A : list, circuit : Circuit, N_M: int):
        '''
        Given a circuit of d qubits that represents a system with two subsystems A and B, 
        it returns the estimated n-Renyi entropy of the system, making N_M measurements. 

        Parameters:
            n: order of the entropy wished to mesure.
            state_size: list of qubits that form the initial state. Must be size 2; first element indicates number of qubits for subspace A 
            and second elements the number of qubits for subspace B.
            circuit: initial system that represents the desired state.
        '''
        circuit1 = circuit
        circuit2 = Circuit(n*d + 1)
        for l in range(n):
            circuit2.add(circuit1.on_qubits(*range(l*d + 1, (l+1)*d + 1)))
        c = circuit2
        c.add(gates.H(0))
        for l in range(n-1):
            for a in N_A:
                c.add(gates.SWAP(l*d + a + 1, (l+1)*d + a + 1).controlled_by(0))

        c.add(gates.H(0))

        c.add(gates.M(0))

        result = c(nshots=N_M)
        freq = result.frequencies()
        prob_0 = freq['0']/N_M
        err_0 = np.sqrt(prob_0*(1 - prob_0)/N_M)
        prob_1 = freq['1']/N_M
        err_1 = np.sqrt(prob_1*(1 - prob_1)/N_M)
        R_n = np.abs(prob_0 - prob_1)
        err_R = np.sqrt(err_0**2 + err_1**2)
        S_n = -np.log2(R_n)
        S_2_error = err_R/(R_n*np.log(2))
        self.entropy_value = S_n
        self.entropy_error = S_2_error
        self.freq_list = freq

    def entropy(self):
        '''
        It returns the estimated entropy.
        '''
        return self.entropy_value
    
    def error(self):
        '''
        It returns the estimated standard error of the entropy estimation for the swap protocol.
        '''
        return self.entropy_error
    
    def frequencies(self, prob = None):
        '''
        It returns the outcomes frequencies of the protocol.
        Parameters:
            prob: probabiliy 0 or 1. If not specified, returns the whole dictionary with both outcomes.
        '''
        if prob is None:
            return self.freq_list
        else:
            return self.freq_list[f'{prob}']
    





    

    

    
