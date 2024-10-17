import pickle
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

fb_list = []
sfb_list = []
em_list = []
for i in range(5):

    with open("exp_results_fb_dim_2_nexp_" + str(i) + ".pickle", "rb") as f:
        fb = pickle.load(f)

    with open("exp_results_semi_fb_dim_2_nexp_" + str(i) + ".pickle", "rb") as g:
        sfb = pickle.load(g)

    with open("exp_results_EM_dim_2_nexp_" + str(i) + ".pickle", "rb") as g:
        em = pickle.load(g)

    fb_list.append(fb["kl_train_process"])
    sfb_list.append(sfb["kl_train_process"])
    em_list.append(em["kl_train_last"])

fb_list = np.array(fb_list)
sfb_list = np.array(sfb_list)
em_list = np.array(em_list)

fb_mean = np.mean(fb_list, axis=0)
fb_std = np.std(fb_list, axis=0)
sfb_mean = np.mean(sfb_list, axis=0)
sfb_std = np.std(sfb_list, axis=0)
em_mean = np.mean(em_list)
em_mean = [em_mean] * fb_mean.shape[0]

xaxis = np.arange(fb_mean.shape[0])
start_iter = 0


# Plot---------------------------------
fig, ax = plt.subplots()

ax.plot(
    xaxis[start_iter:],
    em_mean[start_iter:],
    label="ULA",
    linestyle="dashed",
    color="limegreen",
    linewidth=2,
)

ax.plot(
    xaxis[start_iter:],
    fb_mean[start_iter:],
    label="FB Euler",
    color="#FF5733",
    linewidth=2,
)
# plt.fill_between(
#     xaxis[start_iter:],
#     fb_mean[start_iter:] - 2 * fb_std[start_iter:],
#     fb_mean[start_iter:] + 2 * fb_std[start_iter:],
#     color="#FF5733",
#     alpha=0.1,
# )
ax.plot(
    xaxis[start_iter:],
    sfb_mean[start_iter:],
    label="semi FB Euler",
    color="#00008B",
    linewidth=2,
)
# plt.fill_between(
#     xaxis[start_iter:],
#     sfb_mean[start_iter:] - 2 * sfb_std[start_iter:],
#     sfb_mean[start_iter:] + 2 * sfb_std[start_iter:],
#     color="#00008B",
#     alpha=0.1,
# )

# plot background curves
for i in range(5):
    ax.plot(xaxis[start_iter:], fb_list[i, :][start_iter:], color="#FF5733", alpha=0.15)
    ax.plot(
        xaxis[start_iter:], sfb_list[i, :][start_iter:], color="#00008B", alpha=0.15
    )

ax.set_xlabel("Number of iterations", fontsize=16)
ax.set_ylabel("KL divergence", fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ax.legend(fontsize=16)
plt.yscale("log")
plt.tight_layout()

# Define the region for the zoomed-in plot
x1, x2, y1, y2 = 30, 40, 5.5 * 1e-2, 7 * 1e-2  # Region of interest (x1, x2, y1, y2)

# Create an inset axes with a zoom factor
axins = zoomed_inset_axes(ax, 5, loc="lower left")
axins.plot(
    xaxis[start_iter:],
    em_mean[start_iter:],
    linestyle="dashed",
    color="limegreen",
    linewidth=2,
)
axins.plot(
    xaxis[start_iter:],
    fb_mean[start_iter:],
    color="#FF5733",
    linewidth=2,
)
axins.plot(
    xaxis[start_iter:],
    sfb_mean[start_iter:],
    color="#00008B",
    linewidth=2,
)
for i in range(5):
    axins.plot(
        xaxis[start_iter:], fb_list[i, :][start_iter:], color="#FF5733", alpha=0.15
    )
    axins.plot(
        xaxis[start_iter:], sfb_list[i, :][start_iter:], color="#00008B", alpha=0.15
    )

# Set limits for the zoomed region
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
ax.indicate_inset_zoom(axins)
axins.set_aspect(aspect=25)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="black")
plt.xticks([])
plt.yticks(fontsize=15)
plt.yscale("log")
# plt.show()
plt.savefig("KL_comparision.pdf")
