import numpy as np

# -------------------------------------------------
D2R = np.pi / 180.0
R2D = 180.0 / np.pi

# -------------------------------------------------
print('---------------------------------')
print('Input Cartesian Value (mm & deg)')
print('---------------------------------')
# Rainbow Robotics use ZYX Euler notation
# Z -> Y' -> X''
input_x = -156.76
input_y = -155.15
input_z = 814.96

input_rx = -43.47
input_ry = 80.56
input_rz = -60.88

# -------------------------------------------------
x = input_x
y = input_y
z = input_z
rx = input_rx * D2R
ry = input_ry * D2R
rz = input_rz * D2R

# -------------------------------------------------
# Link Length parameter (RB5 - 850)
d1 = 169.2
d2 = 148.4
d3 = 148.4
d4 = 110.7
d5 = 110.7
d6 = 96.7
a1 = 425.0
a2 = 392.0

# -------------------------------------------------
Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
               [np.sin(rz), np.cos(rz), 0],
               [0, 0, 1]])

Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
               [0, 1, 0],
               [-np.sin(ry), 0, np.cos(ry)]])

Rx = np.array([[1, 0, 0],
               [0, np.cos(rx), -np.sin(rx)],
               [0, np.sin(rx), np.cos(rx)]])

R = np.dot(np.dot(Rz, Ry), Rx)

# -------------------------------------------------
Y06 = R[:, 1]
P06 = np.array([x, y, z])
P05 = P06 + d6 * Y06
th1 = np.arctan2(P05[1], P05[0]) - np.arccos(d4 / np.sqrt(P05[1]**2 + P05[0]**2)) + 0.5 * np.pi
th5 = np.arccos((np.sin(th1) * P06[0] - np.cos(th1) * P06[1] - d4) / d6)
th6 = np.arctan2(-(-np.sin(th1) * R[0, 0] + np.cos(th1) * R[1, 0]) / np.sin(th5), (-np.sin(th1) * R[0, 2] + np.cos(th1) * R[1, 2]) / np.sin(th5)) + 0.5 * np.pi

A01 = np.array([[np.cos(th1), -np.cos(-0.5 * np.pi) * np.sin(th1), np.sin(-0.5 * np.pi) * np.sin(th1), 0],
                [np.sin(th1), np.cos(-0.5 * np.pi) * np.cos(th1), -np.sin(-0.5 * np.pi) * np.cos(th1), 0],
                [0, np.sin(-0.5 * np.pi), np.cos(-0.5 * np.pi), d1],
                [0, 0, 0, 1]])

A67 = np.array([[np.cos(0), -np.cos(0.5 * np.pi) * np.sin(0), np.sin(0.5 * np.pi) * np.sin(0), 0],
                [np.sin(0), np.cos(0.5 * np.pi) * np.cos(0), -np.sin(0.5 * np.pi) * np.cos(0), 0],
                [0, np.sin(0.5 * np.pi), np.cos(0.5 * np.pi), 0],
                [0, 0, 0, 1]])

A78 = np.array([[np.cos(th5), -np.cos(-0.5 * np.pi) * np.sin(th5), np.sin(-0.5 * np.pi) * np.sin(th5), 0],
                [np.sin(th5), np.cos(-0.5 * np.pi) * np.cos(th5), -np.sin(-0.5 * np.pi) * np.cos(th5), 0],
                [0, np.sin(-0.5 * np.pi), np.cos(-0.5 * np.pi), d5],
                [0, 0, 0, 1]])

A89 = np.array([[np.cos(th6), -np.cos(0.5 * np.pi) * np.sin(th6), np.sin(0.5 * np.pi) * np.sin(th6), 0],
                [np.sin(th6), np.cos(0.5 * np.pi) * np.cos(th6), -np.sin(0.5 * np.pi) * np.cos(th6), 0],
                [0, np.sin(0.5 * np.pi), np.cos(0.5 * np.pi), -d6],
                [0, 0, 0, 1]])

A17 = np.linalg.inv(A01) @ np.vstack((np.hstack((R, P06.reshape(3, 1))), [0, 0, 0, 1])) @ np.linalg.inv(A89) @ np.linalg.inv(A78) @ np.linalg.inv(A67)

P14 = A17[:3, 3]
th3 = np.arccos((P14[0]**2 + P14[1]**2 - a1**2 - a2**2) / (2 * a1 * a2))
th2 = np.arctan2(P14[0], -P14[1]) - np.arcsin(a2 * np.sin(th3) / np.sqrt(P14[0]**2 + P14[1]**2))

A12 = np.array([[np.cos(th2 - 0.5 * np.pi), -np.cos(0) * np.sin(th2 - 0.5 * np.pi), np.sin(0) * np.sin(th2 - 0.5 * np.pi), 0],
                [np.sin(th2 - 0.5 * np.pi), np.cos(0) * np.cos(th2 - 0.5 * np.pi), -np.sin(0) * np.cos(th2 - 0.5 * np.pi), 0],
                [0, np.sin(0), np.cos(0), -d2],
                [0, 0, 0, 1]])

A23 = np.array([[np.cos(0), -np.cos(0) * np.sin(0), np.sin(0) * np.sin(0), a1 * np.cos(0)],
                [np.sin(0), np.cos(0) * np.cos(0), -np.sin(0) * np.cos(0), a1 * np.sin(0)],
                [0, np.sin(0), np.cos(0), 0],
                [0, 0, 0, 1]])

A34 = np.array([[np.cos(th3), -np.cos(0) * np.sin(th3), np.sin(0) * np.sin(th3), 0],
                [np.sin(th3), np.cos(0) * np.cos(th3), -np.sin(0) * np.cos(th3), 0],
                [0, np.sin(0), np.cos(0), d3],
                [0, 0, 0, 1]])

A45 = np.array([[np.cos(0), -np.cos(0) * np.sin(0), np.sin(0) * np.sin(0), a2 * np.cos(0)],
                [np.sin(0), np.cos(0) * np.cos(0), -np.sin(0) * np.cos(0), a2 * np.sin(0)],
                [0, np.sin(0), np.cos(0), 0],
                [0, 0, 0, 1]])

A56_cal = np.linalg.inv(A45) @ np.linalg.inv(A34) @ np.linalg.inv(A23) @ np.linalg.inv(A12) @ np.linalg.inv(A01) @ np.vstack((np.hstack((R, P06.reshape(3, 1))), [0, 0, 0, 1])) @ np.linalg.inv(A89) @ np.linalg.inv(A78) @ np.linalg.inv(A67)
th4 = np.arctan2(A56_cal[1, 0], A56_cal[0, 0]) - 0.5 * np.pi

# -------------------------------------------------
print('---------------------------------')
print('Inverse Kinematics Result (deg)')
print('---------------------------------')
th1 = th1 * R2D
th2 = th2 * R2D
th3 = th3 * R2D
th4 = th4 * R2D
th5 = th5 * R2D
th6 = th6 * R2D

print(f'th1: {th1:.2f} degrees')
print(f'th2: {th2:.2f} degrees')
print(f'th3: {th3:.2f} degrees')
print(f'th4: {th4:.2f} degrees')
print(f'th5: {th5:.2f} degrees')
print(f'th6: {th6:.2f} degrees')
