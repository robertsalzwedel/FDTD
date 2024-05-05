# import matplotlib.pyplot as plt
# import numpy as np

# np.random.seed(19680801)
# data = np.random.random((50, 50, 50))

# fig, ax = plt.subplots(2,2)

# for i, img in enumerate(data):
#     ax.clear()
#     ax[0,0].imshow(img)
#     ax[0,0].set_title(f"frame {i}")
#     # Note that using time.sleep does *not* work here!
#     plt.pause(0.1)



# import matplotlib.pyplot as plt
# import numpy as np

# fig, axs = plt.subplots(2, 2)
# cmaps = ['RdBu_r', 'viridis']

# for t in range(1000):
#     axs[0,0].clear()
#     axs[0,1].clear()
#     axs[1,0].clear()
#     axs[1,1].clear()
#     fig.clear()
#     for col in range(2):
#         for row in range(2):
#             ax = axs[row, col]
#             img = ax.contourf(np.random.random((20, 20)) )
#             cbar=plt.colorbar(img, ax=ax)
#     plt.pause(0.1)
# #plt.show()

# fig, axs = plt.subplots(2, 2)
# cmaps = ['RdBu_r', 'viridis']

# for t in range(1000):
#     axs[0,0].clear()
#     axs[0,1].clear()
#     axs[1,0].clear()
#     axs[1,1].clear()
#     for col in range(2):
#         #fig.clear()
#         for row in range(2):
#             ax = axs[row, col]
#             ax.imshow(np.random.random((20, 20)) * (col + 1))
# #        fig.colorbar(pcm, ax=axs[:, col], shrink=0.6)
#     plt.pause(0.1)
# #plt.show()#    
# img = ax.imshow(Ez)

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 6*np.pi, 100)
y = np.sin(x)

# You probably won't need this if you're embedding things in a tkinter plot...
#plt.ion()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# line1, = ax.plot(x, y, 'r-') # Returns a tuple of line objects, thus the comma

# for phase in np.linspace(0, 10*np.pi, 500):
#     line1.set_ydata(np.sin(x + phase))
#     fig.canvas.draw()
#     fig.canvas.flush_events()
#     plt.pause(0.01)

size = 20

# plt.rcParams.update({'font.size': 10})
# fig, ax = plt.subplots(1, 1, figsize=(10, 6))  # animation fig
# ims = []
# ims.append(ax.imshow(np.zeros((20, 20)),vmin = 0, vmax = 1))

# cbaxes = fig.add_axes([0.9, 0.58, 0.01, 0.35])
# cbar = plt.colorbar(ims[0], cax=cbaxes)
# cbar.set_label('Field [arb. units]')



plt.rcParams.update({'font.size': 10})
fig, ax = plt.subplots(3, 4, figsize=(16, 9))  # animation fig
ims = []

ims.append(ax[0,1].imshow(np.zeros((1, 1)), cmap='viridis',
                            interpolation='quadric', origin='lower'))

ims.append(ax[0,2].imshow(np.zeros((1, 1)), cmap='viridis',
                            interpolation='quadric', origin='lower'))

ims.append(ax[0,3].imshow(np.zeros((1, 1)), cmap='viridis',
                            interpolation='quadric', origin='lower'))

for im in ims:
    im.set_clim(vmin=0, vmax=1)

cbaxes = fig.add_axes([0.9, 0.58, 0.01, 0.35])
cbar = plt.colorbar(ims[0], cax=cbaxes)
cbar.set_label('Field [arb. units]')

for i in range(1000):
    ims[0].set_data(np.random.random((size, size)))
    ims[1].set_data(np.random.random((size, size)))
    plt.pause(.1)
