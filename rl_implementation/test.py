# import matplotlib.pyplot as plt

# def identify_axes(ax_dict, fontsize=48):
#     kw = dict(ha="center", va="center", fontsize=fontsize, color="darkgrey")
#     for k, ax in ax_dict.items():
#         ax.text(0.5, 0.5, k, transform=ax.transAxes, **kw)


# axd = plt.figure(layout="constrained").subplot_mosaic(
#     """
#     AC
#     BC
#     """
# )
# identify_axes(axd)
# plt.show()

tup = (1, 2)
print(str(tup))