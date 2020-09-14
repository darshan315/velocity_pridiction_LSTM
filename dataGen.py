def generate_data(amplitude, freq):
    import numpy as np

    Exact = []
    vel = []
    ic = 100
    t_final = 20000
    dt = 10
    radius = 10.0
    di = 3.0
    row = []
    for i in range(200):
        a1 = abs(i - ic)
        phi = 0.5 - 0.5 * np.math.tanh(2 * (a1 - radius) / di)
        row.append(phi)
    Exact.append(row)

    for t in range(0, t_final + 1, dt):
        val = amplitude * np.math.cos(freq * t / t_final * np.math.pi)
        #    vel.append(val)
        row = []
        row1 = []
        ic += val * dt
        for i in range(200):
            a1 = abs(i - ic)
            phi = 0.5 - 0.5 * np.math.tanh(2 * (a1 - radius) / di)
            #            phi = - 0.5*np.math.tanh(2*(a1-radius)/di)
            #    Exact.extend([phi])
            row.append(phi)
            row1.append(val)
        Exact.append(row)
        vel.append(row1)

    vel = np.array(vel)
    Exact = np.array(Exact)
    Exact = np.delete(Exact, -1, 0)

    return Exact, vel


def data_preparation(freq, phi):
    import numpy as np
    n_removal = 1
    n_features = 1
    val_i = []
    val_o = []
    for i in range(len(freq)):
        a, v = generate_data(phi[i], freq[i])
        a_i = a[:-n_removal, :]
        a_o = a[n_removal:, :]
        a_i = a_i.reshape(a_i.shape[0], n_features, a_i.shape[1])
        a_o = a_o.reshape(a_o.shape[0], n_features, a_o.shape[1])
        val_i.append(a_i); val_o.append(a_o)
        a_i = np.array(val_i); a_o = np.array(val_o)
    a_i = a_i.reshape(a_i.shape[0] * a_i.shape[1], a_i.shape[2], a_i.shape[3])
    a_o = a_o.reshape(a_o.shape[0] * a_o.shape[1], a_o.shape[2], a_o.shape[3])
    return a_i, a_o

