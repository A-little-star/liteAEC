""" Partitioned-Block-Based Frequency Domain Kalman Filter """

import numpy as np
from numpy.fft import rfft as fft
from numpy.fft import irfft as ifft
import sys

debug = 0
idx = 3

class PFDKF:
    def __init__(self, N, M, A=0.999, P_initial=10):
        self.N = N
        self.M = M
        self.A = A
        self.A2 = A**2
        self.m_soomth_factor = 0.5

        self.x = np.zeros(shape=(2*self.M), dtype=np.float32)

        self.m = np.zeros(shape=(self.M + 1), dtype=np.float32)
        self.P = np.full((self.N, self.M + 1), P_initial)
        self.X = np.zeros((self.N, self.M + 1), dtype=complex)
        self.H = np.zeros((self.N, self.M + 1), dtype=complex)
        self.mu = np.zeros((self.N, self.M + 1), dtype=complex)
        self.half_window = np.concatenate(([1]*self.M, [0]*self.M))


    def filt(self, x, d):
        assert(len(x) == self.M)
        self.x = np.concatenate([self.x[self.M:], x])
        X = fft(self.x)
        self.X[1:] = self.X[:-1]
        self.X[0] = X

        if debug == idx:
            output_file = '/home/node25_tmpdata/xcli/percepnet/c_aec/test_txt/H_last_py.txt'
            with open(output_file, 'w') as f:
                for i in range(0, len(self.H[0]), 8):
                    # 获取当前行的100个数
                    line = ' '.join(f"{value:.6f}" for value in self.H[1].imag[i:i+8])
                    f.write(line + '\n')
        
        if debug == idx:
            output_file = '/home/node25_tmpdata/xcli/percepnet/c_aec/test_txt/X.txt'
            with open(output_file, 'w') as f:
                for i in range(0, len(self.X[0]), 8):
                    # 获取当前行的100个数
                    line = ' '.join(f"{value:.6f}" for value in self.X[0].real[i:i+8])
                    f.write(line + '\n')

        Y = np.sum(self.H*self.X, axis=0)

        if debug == idx:
            output_file = '/home/node25_tmpdata/xcli/percepnet/c_aec/test_txt/Y_py.txt'
            with open(output_file, 'w') as f:
                for i in range(0, len(Y), 8):
                    # 获取当前行的100个数
                    line = ' '.join(f"{value:.6f}" for value in Y.real[i:i+8])
                    f.write(line + '\n')

        y = ifft(Y).real[self.M:]
        e = d-y

        if debug == idx:
            output_file = '/home/node25_tmpdata/xcli/percepnet/c_aec/test_txt/d_py.txt'
            with open(output_file, 'w') as f:
                for i in range(0, len(d), 8):
                    # 获取当前行的100个数
                    line = ' '.join(f"{value:.6f}" for value in d[i:i+8])
                    f.write(line + '\n')
        if debug == idx:
            output_file = '/home/node25_tmpdata/xcli/percepnet/c_aec/test_txt/e_mid_py.txt'
            with open(output_file, 'w') as f:
                for i in range(0, len(e), 8):
                    # 获取当前行的100个数
                    line = ' '.join(f"{value:.6f}" for value in e[i:i+8])
                    f.write(line + '\n')

        e_fft = np.concatenate((np.zeros(shape=(self.M,), dtype=np.float32), e))
        self.E = fft(e_fft)
        X2 = np.sum(np.abs(self.X)**2, axis=0)

        if debug == idx:
            output_file = '/home/node25_tmpdata/xcli/percepnet/c_aec/test_txt/E_py.txt'
            with open(output_file, 'w') as f:
                for i in range(0, len(self.E), 8):
                    # 获取当前行的100个数
                    line = ' '.join(f"{value:.6f}" for value in self.E.real[i:i+8])
                    f.write(line + '\n')

        if debug == idx:
            output_file = '/home/node25_tmpdata/xcli/percepnet/c_aec/test_txt/m_last_py.txt'
            with open(output_file, 'w') as f:
                for i in range(0, len(self.m), 8):
                    # 获取当前行的100个数
                    line = ' '.join(f"{value:.6f}" for value in self.m[i:i+8])
                    f.write(line + '\n')

        self.m = self.m_soomth_factor * self.m + (1-self.m_soomth_factor) * np.abs(self.E)**2
        R = np.sum(self.X*self.P*self.X.conj(), 0) + 2*self.m/self.N

        if debug == idx:
            output_file = '/home/node25_tmpdata/xcli/percepnet/c_aec/test_txt/m_py.txt'
            with open(output_file, 'w') as f:
                for i in range(0, len(self.m), 8):
                    # 获取当前行的100个数
                    line = ' '.join(f"{value:.6f}" for value in self.m[i:i+8])
                    f.write(line + '\n')

        if debug == idx:
            output_file = '/home/node25_tmpdata/xcli/percepnet/c_aec/test_txt/R_py.txt'
            with open(output_file, 'w') as f:
                for i in range(0, len(R), 8):
                    # 获取当前行的100个数
                    line = ' '.join(f"{value:.6f}" for value in R.real[i:i+8])
                    f.write(line + '\n')

        if debug == idx:
            output_file = '/home/node25_tmpdata/xcli/percepnet/c_aec/test_txt/P_py.txt'
            with open(output_file, 'w') as f:
                for i in range(0, len(self.P[0]), 8):
                    # 获取当前行的100个数
                    line = ' '.join(f"{value:.6f}" for value in self.P[1].real[i:i+8])
                    f.write(line + '\n')

        self.mu = self.P / (R + 1e-10)
        W = 1 - np.sum(self.mu*np.abs(self.X)**2, 0)
        E_res = W*self.E
        e = ifft(E_res).real[self.M:].real
        y = d-e

        if debug == idx:
            output_file = '/home/node25_tmpdata/xcli/percepnet/c_aec/test_txt/e_py.txt'
            with open(output_file, 'w') as f:
                for i in range(0, len(e), 8):
                    # 获取当前行的100个数
                    line = ' '.join(f"{value:.6f}" for value in e[i:i+8])
                    f.write(line + '\n')

        return e, y

    def update(self):
        G = self.mu*self.X.conj()
        if debug == idx:
            output_file = '/home/node25_tmpdata/xcli/percepnet/c_aec/test_txt/G_py.txt'
            with open(output_file, 'w') as f:
                for i in range(0, len(G[0]), 8):
                    # 获取当前行的100个数
                    line = ' '.join(f"{value:.6f}" for value in G[1].real[i:i+8])
                    f.write(line + '\n')
        if debug == 4:
            print(self.mu[0])
        self.P = self.A2*(1 - 0.5*G*self.X)*self.P + (1-self.A2)*np.abs(self.H)**2
        self.H = self.A*(self.H + fft(self.half_window*(ifft(self.E*G).real)))

        # if debug == 4:
        #     print(self.H[2])

        if debug == idx:
            output_file = '/home/node25_tmpdata/xcli/percepnet/c_aec/test_txt/X_py.txt'
            with open(output_file, 'w') as f:
                for i in range(0, len(self.X[0]), 8):
                    # 获取当前行的100个数
                    line = ' '.join(f"{value:.6f}" for value in self.X[0].real[i:i+8])
                    f.write(line + '\n')

        if debug == idx:
            output_file = '/home/node25_tmpdata/xcli/percepnet/c_aec/test_txt/mu_py.txt'
            with open(output_file, 'w') as f:
                for i in range(0, len(self.mu[0]), 8):
                    # 获取当前行的100个数
                    line = ' '.join(f"{value:.6f}" for value in self.mu[1].real[i:i+8])
                    f.write(line + '\n')

        if debug == idx:
            output_file = '/home/node25_tmpdata/xcli/percepnet/c_aec/test_txt/H_py.txt'
            with open(output_file, 'w') as f:
                for i in range(0, len(self.H[0]), 8):
                    # 获取当前行的100个数
                    line = ' '.join(f"{value:.6f}" for value in self.H[1].real[i:i+8])
                    f.write(line + '\n')


def pfdkf(x, d, N=10, M=256, A=0.999, P_initial=10):
    ft = PFDKF(N, M, A, P_initial)
    num_block = min(len(x), len(d)) // M

    e = np.zeros(num_block*M)
    y = np.zeros(num_block*M)
    for n in range(num_block):
        global debug
        debug += 1
        x_n = x[n*M:(n+1)*M]
        d_n = d[n*M:(n+1)*M]
        e_n, y_n = ft.filt(x_n, d_n)
        ft.update()

        # if n == 10:
        #     output_file = '/home/node25_tmpdata/xcli/percepnet/c_aec/test_txt/debug_py.txt'
        #     with open(output_file, 'w') as f:
        #         for i in range(0, len(e_n), 8):
        #             # 获取当前行的100个数
        #             line = ' '.join(f"{value:.6f}" for value in e_n[i:i+8])
        #             f.write(line + '\n')
        #     break


        
        e[n*M:(n+1)*M] = e_n
        y[n*M:(n+1)*M] = y_n
    return e, y

def process(ref_path, mic_path, sr=16000):
    import soundfile as sf
    mic ,sr= sf.read(mic_path)
    ref ,sr= sf.read(ref_path)
    mic = mic.astype(np.float32)*100
    ref = ref.astype(np.float32)*100
    error, echo = pfdkf(ref, mic, N=10, M=400, A=0.999, P_initial=10)
    sf.write('./test_wav/e_py.wav', error/100, sr)
    sf.write('./test_wav/y_py.wav', echo/100, sr)
    print(f'mic: {mic.shape}')
    print(f'ref: {ref.shape}')
    print(f'e: {error.shape}')
    print(f'y: {echo.shape}')
    print(f'type: {mic.dtype}')

    # output_file = '/home/node25_tmpdata/xcli/percepnet/c_aec/test_txt/e_py.txt'

    # with open(output_file, 'w') as f:
    #     for i in range(0, len(error), 100):
    #         # 获取当前行的100个数
    #         line = ' '.join(f"{value:.6f}" for value in error[i:i + 100])
    #         f.write(line + '\n')

def frame_process(ref_path, mic_path):
    import soundfile as sf
    mic ,sr= sf.read(mic_path)
    ref ,sr= sf.read(ref_path)
    mic = mic.astype(np.float32)*100
    ref = ref.astype(np.float32)*100
    length = mic.shape[-1]
    error_list = []
    echo_list = []
    block = 4000
    num_blocks = length // block
    for i in range(num_blocks):
        e, y = pfdkf(ref[i*block:(i+1)*block], mic[i*block:(i+1)*block], N=10, M=400, A=0.999, P_initial=10)
        error_list.append(e)
        echo_list.append(y)
    error = np.concatenate(error_list, axis=0)
    echo = np.concatenate(echo_list, axis=0)
    print(error.shape)

    sf.write('./test_wav/e_py.wav', error/100, sr)
    sf.write('./test_wav/y_py.wav', echo/100, sr)
    print(f'mic: {mic.shape}')
    print(f'ref: {ref.shape}')
    print(f'e: {error.shape}')
    print(f'y: {echo.shape}')
    print(f'type: {mic.dtype}')

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=6)
    mic = '/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/mic.wav'
    ref = '/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/ref.wav'
    process(ref, mic)