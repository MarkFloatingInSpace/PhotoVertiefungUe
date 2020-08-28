import numpy as np
import matplotlib.pyplot as plt

sensor_size = (3000, 4000)

x = np.arange(-sensor_size[0]/2, sensor_size[0]/2, 1)
y = np.arange(-sensor_size[1]/2, sensor_size[1]/2, 1)

xx, yy = np.meshgrid(x, y)
rho = (xx**2 + yy**2) ** (1/2)

rho0 = (2000 ** 2 + 1500 ** 2) ** (1 / 2) /2

k1, k2 = (+1.08099e-08, -2.41552e-15)

#delta_x_dist = xx * (k1 * (rho - rho0) ** 2 + k2 * (rho - rho0) ** 4 + k3 * (rho - rho0) ** 6) + \
         #      2 * p1 * xx * yy + p2 * (rho ** 2 + 2 * xx ** 2)
#delta_y_dist = yy * (k1 * (rho - rho0) ** 2 + k2 * (rho - rho0) ** 4 + k3 * (rho - rho0) ** 6) + \
         #      + p1 * (rho ** 2 + 2 * yy ** 2) + 2 * p2 * xx * yy

#NUR radiale Verzeichnung
delta_x_dist = xx *(k1 * (rho**2-rho0**2) + k2 * (rho**4-rho0**4))
delta_y_dist = yy * (k1 * (rho**2-rho0**2) + k2 * (rho**4-rho0**4))

#delta_x_dist = xx*(k1*(rho**2-rho0**2) + k2 * (rho ** 4-rho0**2)) + 2 * p1 * xx * yy + p2 * (rho ** 2 + 2 * xx ** 2)
#delta_y_dist = yy*(k1*(rho**2-rho0**2) + k2 * (rho ** 4-rho0**2)) + p1 * (rho ** 2 + 2 * yy ** 2) + 2 * p2 * xx * yy

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