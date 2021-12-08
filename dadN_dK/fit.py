import numpy as np
import matplotlib.pyplot as plt

PIXEL = 6.1e-6  # m
dS = 200  # MPa

# shape correction factor matrix
M = [[[1.095, 0.113, -0.896], [-1.336, 1.824, 3.092], [13.108, -21.709, -4.197], [-43.689, 105.483, -13.255], [134.868, -271.225, 51.548], [-242.653, 387.47, -59.329], [254.093, -290.024, 13.481], [-108.196, 88.387, 10.854]],
     [[-1.177, 0.271, 0.904], [17.924, -11.649, 0.701], [-137.252, 98.358, -32.641], [545.816, -415.027, 204.104], [-1223.334,
                                                                                                                    982.713, -568.407], [1541.587, -1329.634, 857.543], [-1006.656, 961.893, -657.659], [264.206, -288.565, 191.57]],
     [[0.725, -0.388, 0.008], [-17.427, 10.074, -4.883], [134.652, -80.088, 55.092], [-551.902, 328.165, -305.079], [1239.493, -772.921, 916.962], [-1548.537, 1055.952, -1545.428], [969.388, -784.581, 1372.595], [-227.132, 245.798, -485.556]]]


def calc_dK(a, h, D, x):
    '''
    Calculate the dK given the crack geometry
    Input:
        x, position of the interested point
    '''
    # compute b
    b = 2*a*h/np.sqrt(-2*D*D+2*D*np.sqrt(D*D-4*h*h)+4*a*a+4*h*h)

    a_b = a/b
    a_D = a/D
    x_h = x/h

    # compute the shape correction factor
    F = 0
    for i in range(len(M)):
        for j in range(len(M[0])):
            for k in range(len(M[0][0])):
                F += M[i][j][k]*(a_b**i)*(a_D**j)*(x_h**k)
    # compute the stress intensity factor
    dK = F*dS*np.sqrt(np.pi*a)
    return dK


def poly_fitting(N, a, hdouble):
    '''
    Fit the N-a (and N-h) curve using incremental polynominal method (2nd order), and
    compute the dadN value using fitted curve.
    Input: 
        N, list of cycle number
        a, crack length
        hdouble, 2*h
    '''
    a_hat = []
    h_hat = []
    dadN = []
    npt = len(N)-4  # number of fitted dadN point
    if(npt < 0):
        return a_hat, h_hat, [], dadN

    for i in range(npt):
        N5 = np.array(N[i:i+5])
        C1 = 0.5*(N5[-1]+N5[0])
        C2 = 0.5*(N5[-1]-N5[0])

        n5 = (N5-C1)/C2  # normalized cycle number
        a5 = np.array(a[i:i+5])*PIXEL
        h5 = 0.5*np.array(hdouble[i:i+5])*PIXEL
        (b2_a, b1_a, b0_a) = np.polyfit(n5, a5, 2)  # b0 + b1*n + b2*n^2
        (b2_h, b1_h, b0_h) = np.polyfit(n5, h5, 2)  # b0 + b1*n + b2*n^2
        # crack growth rate
        dadN.append(b1_a/C2+2*b2_a*n5[2]/C2)
        # fitted a and h
        a_hat.append(b0_a + b1_a*n5[2] + b2_a*n5[2]*n5[2])
        h_hat.append(b0_h + b1_h*n5[2] + b2_h*n5[2]*n5[2])
    return a_hat, h_hat, N[2:-2], dadN


def power_fitting(dK, dadN):
    '''
    Fit the dadN-dK curve using power law, i.e., dadN=a*dK^b
    Output: 
        a, b: fitting parameters
    '''
    log_dadN = np.log(dadN)
    log_dK = np.log(dK)
    (b, log_a) = np.polyfit(log_dK, log_dadN,  1)
    return np.exp(log_a), b

# test the curve fitting function
# _, a_hat, h_hat, dadN = poly_fitting(N, a, hdouble)
# a, b = power_fitting(N[2:-2], dadN)
# plt.plot(N[2:-2], a*N[2:-2]**b)
# plt.plot(N[2:-2], dadN)


sample = "AMY6a"  # AMS2, AMG5, AMG6a, AMY6, AMY6a
dadN_tot = []
dK_tot = []
N_tot = []
a_tot = []
h_tot = []
with open(sample+".txt") as f:
    lines = f.read().split('\n')
    D = int(lines[0])
    N, a, hdouble = [], [], []
    for line in lines[1:]:
        if(N == [] and line[0] == '#'):
            continue
        elif(N != [] and line[0] == '#'):
            a_hat, h_hat, N_cut, dadN = poly_fitting(N, a, hdouble)
            dK = []
            for i in range(len(a_hat)):
                dK.append(calc_dK(a_hat[i], h_hat[i], D, 0))
            a_tot.append(a_hat)
            h_tot.append(h_hat)
            N_tot.append(N_cut)
            dadN_tot.append(dadN)
            dK_tot.append(dK)
            N, a, hdouble = [], [], []
        else:
            data = line.split()
            N.append(int(data[0]))
            a.append(int(data[1]))
            hdouble.append(int(data[2]))

# print(len(dadN_tot), len(dK_tot), len(N_tot), len(a_tot), len(h_tot))
# output data into txt file
# a, h, N, dadN, dK
out = open(sample+"_data.txt", "a")
for i in range(len(N_tot)):
    out.write('#'+str(i+1)+'\n')
    out.write("{:>16} {:>16} {:>10} {:>10} {:>10} \n".format(
        "a", "h", "N", "dadN", "dK"))
    for k in range(len(N_tot[i])):
        out.write("{:>16.6} {:>16.6} {:>10} {:>10.6} {:>10.6} \n".format(a_tot[i][k], h_tot[i][k], N_tot[i]
                  [k], dadN_tot[i][k], dK_tot[i][k]))
out.close()

plt.show()
