from scipy import signal

def disc(A,B,C,D,Ts):
    sys_cont = signal.StateSpace(A, B, C, D)
    sys_disc = sys_cont.to_discrete(Ts, method='zoh')
    A = sys_disc.A
    B = sys_disc.B
    C = sys_disc.C
    D = sys_disc.D
    return A, B, C, D