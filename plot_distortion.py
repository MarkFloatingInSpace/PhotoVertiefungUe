import numpy as np
import matplotlib.pyplot as plt

sensor_size = (2976, 3968)

x = np.arange(-sensor_size[0]/2, sensor_size[0]/2, 1)
y = np.arange(-sensor_size[1]/2, sensor_size[1]/2, 1)

xx, yy = np.meshgrid(x, y)
rho = (xx**2 + yy**2) ** (1/2)

rho0 = (3000 ** 2 + 2000 ** 2) ** (1 / 2)

# old
k1, k2, k3, p1, p2 = [+8.65434e-09, -2.94514e-15, +2.99218e-22, -4.57158e-07, -3.01102e-07]

# k1, k2, p1, p2 = [+8.61915e-09, -7.89278e-16, +2.58932e-08, -3.59427e-07]

#
delta_x_dist = xx * (k1 * (rho - rho0) ** 2 + k2 * (rho - rho0) ** 4 + k3 * (rho - rho0) ** 6) + \
               2 * p1 * xx * yy + p2 * (rho ** 2 + 2 * xx ** 2)

# k3 * (rho - rho0) ** 6) + \
delta_y_dist = yy * (k1 * (rho - rho0) ** 2 + k2 * (rho - rho0) ** 4 + k3 * (rho - rho0) ** 6) + \
               + p1 * (rho ** 2 + 2 * yy ** 2) + 2 * p2 * xx * yy

# plt.matshow(delta_x_dist, cmap="PRGn")
# plt.colorbar()
#
# plt.matshow(delta_y_dist, cmap="PRGn")
# plt.colorbar()

total_correction = (delta_x_dist ** 2 + delta_y_dist ** 2) ** (1/2)
plt.matshow(total_correction)
cb = plt.colorbar()
cb.set_label("Betrag der Verzeichnungskorrektur [px]")

# plt.figure()
# plt.quiver(xx, yy, delta_x_dist, delta_y_dist)

plt.show()