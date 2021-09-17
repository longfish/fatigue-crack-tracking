'''
Calculate the shape correction factor of the round bar.
'''

M = [[[1.095, 0.113, -0.896], [-1.336, 1.824, 3.092], [13.108, -21.709, -4.197], [-43.689, 105.483, -13.255], [134.868, -271.225, 51.548], [-242.653, 387.47, -59.329], [254.093, -290.024, 13.481], [-108.196, 88.387, 10.854]],
     [[-1.177, 0.271, 0.904], [17.924, -11.649, 0.701], [-137.252, 98.358, -32.641], [545.816, -415.027, 204.104], [-1223.334,
                                                                                                                    982.713, -568.407], [1541.587, -1329.634, 857.543], [-1006.656, 961.893, -657.659], [264.206, -288.565, 191.57]],
     [[0.725, -0.388, 0.008], [-17.427, 10.074, -4.883], [134.652, -80.088, 55.092], [-551.902, 328.165, -305.079], [1239.493, -772.921, 916.962], [-1548.537, 1055.952, -1545.428], [969.388, -784.581, 1372.595], [-227.132, 245.798, -485.556]]]


def calc_F(a_b, a_D, x_h):
    F = 0
    for i in range(len(M)):
        for j in range(len(M[0])):
            for k in range(len(M[0][0])):
                F += M[i][j][k]*(a_b**i)*(a_D**j)*(x_h**k)
    return F


'''
# print out the M matrix
for j in range(len(M[0])):
    for i in range(len(M)):
        print('{:-9}'.format(M[i][j][2]), end=' ')
    print()
'''

# custom geometry parameters
a_b = [0.472426,
       0.543408,
       0.550589,
       0.557785,
       0.579458,
       0.615859,
       0.652595
       ]
a_D = [0.043131,
       0.055324,
       0.056631,
       0.057955,
       0.062022,
       0.069125,
       0.076635
       ]
for i in range(len(a_b)):
    print(
        '{:-9} {:-9}'.format(calc_F(a_b[i], a_D[i], 0), calc_F(a_b[i], a_D[i], 1)))
print()

# print(calc_F(1, 0.6, 0))
