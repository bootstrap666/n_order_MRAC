import numpy as np
from scipy.linalg import expm

class linear_system:
    def __init__(self,b:np.ndarray,a:np.ndarray,sampling_rate:float):

        if (len(b) > len(a)):
            print("improper transfer function", file=sys.stderr)
            
        if (len(b) == len(a)):
            self._D, b = np.polydiv(b, a)
        else:
            self._D = 0
        
        self._N = len(a)-1
        self._M = len(b)
        self._Ts = 1.0/sampling_rate
        
        a = a/a[0]
        b = b/a[0]
        
        # Initializing the state-space matrices in the controllability canonical form
        # Matrix A
        self._A = np.zeros((self._N,self._N))
        self._A[0:-1,1:] = np.eye(self._N-1)
        self._A[-1,:] = -a[-1:0:-1]
        
        # Matrix B
        self._B = np.zeros(self._N)
        self._B[-1] = 1

        # Matrix C
        self._C = np.zeros(self._N)
        self._C[:self._M] = b[::-1] 

    def iterate(self, input_signal_sample):
        self._X = (np.eye(self._N) + self._Ts*self._A)@self._X + self._Ts*self._B*[input_signal_sample]
        y = self._C@self._X + self._D*input_signal_sample
        return y

    def simulate(self, input_signal, initial_conditions):
        n_iter = len(input_signal)

        state_history = np.zeros((self._N, n_iter))
        output_history = np.zeros(n_iter)

        self._X = initial_conditions
        yold = 0
        for i in range(n_iter):
            state_history[:,i] = self._X
            output_history[i] = yold 
            yold = self.iterate(input_signal[i])
            
        return output_history, state_history
        
    def print(self):
        print(f"A={self._A}")
        print(f"B={self._B}")
        print(f"C={self._C}")
        print(f"D={self._D}")

    def set_null_state(self):
        self._X = np.zeros(self._N)
        
    def states(self):
        return self._X

    def print_zoh(self):
        print(f"A_d={expm(self._Ts*self._A)}")
        print(f"B_d={np.linalg.solve(self._A, ((expm(self._Ts*self._A))-np.eye(self._N))@self._B)}")
        print(f"C_d={self._C}")
        print(f"D_d={self._D}")

    
    def print_Euler(self):
        print(f"A_d={(np.eye(self._N) + self._Ts*self._A)}")
        print(f"B_d={self._Ts*self._B}")
        print(f"C_d={self._C}")
        print(f"D_d={self._D}")
        print(f"\lambda = {np.linalg.eig(np.eye(self._N) + self._Ts*self._A)}")

    def iterate_zoh(self, input_signal_sample):
        self._X = expm(self._Ts*self._A)@self._X + np.linalg.solve(self._A, ((expm(self._Ts*self._A))-np.eye(self._N))@self._B)*[input_signal_sample]
        y = self._C@self._X + self._D*input_signal_sample
        return y

    def simulate_zoh(self, input_signal, initial_conditions):
        n_iter = len(input_signal)

        state_history = np.zeros((self._N, n_iter))
        output_history = np.zeros(n_iter)

        self._X = initial_conditions
        yold = 0
        for i in range(n_iter):
            state_history[:,i] = self._X
            output_history[i] = yold 
            yold = self.iterate_zoh(input_signal[i])
            
        return output_history, state_history
