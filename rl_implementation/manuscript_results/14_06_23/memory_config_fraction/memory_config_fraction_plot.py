import matplotlib.pyplot as plt
import numpy as np
import os
import json
import matplotlib.patches as mplpatches
from matplotlib.lines import Line2D

################### CUSTOM BOX PLOT ###################
def artisinal_boxplot(yvalues, pos, width, color, panel, median_line_color = "black", median_linewidth = 1):
    P50 = np.median(yvalues) # it's the median
    P5 = np.percentile(yvalues,5) # 5% of data values in this distribution are below P5
    P25 = np.percentile(yvalues,25) # 25% of data values in this distribution are below P5
    P75 = np.percentile(yvalues,75) # 75% of data values in this distribution are below P5
    P95 = np.percentile(yvalues,95) # 95% of data values in this distribution are below P5

    left = pos - (width/2) # pos variable will be the center of boxplot
    bottom = P25
    width = width
    height = P75 - P25

    rectangle = mplpatches.Rectangle([left,bottom], width, height,
                                    facecolor = color,
                                    edgecolor = 'black',
                                    linewidth = 0.75, # width of the edge around rectangle
                                    )
    panel.add_patch(rectangle)
    ## Drawing the median line
    panel.plot([pos-(width/2), pos+(width/2)], [P50, P50], 
            marker = 'o',
            markerfacecolor = 'red',
            markeredgewidth = 0, # width of line around marker point, '0' for no marker width
            markersize = 0,
            color = median_line_color, # line color
            linewidth = median_linewidth, # line width, '0' for no line
            alpha = 1) # opacity
    
    ## Drawing vertical lines joining 75th and 95th percentile and 25th and 5th percentile
    panel.plot([pos, pos], [P75, P95], 
            marker = 'o',
            markerfacecolor = 'red',
            markeredgewidth = 0, # width of line around marker point, '0' for no marker width
            markersize = 0,
            color = 'black', # line color
            linewidth = 1, # line width, '0' for no line
            alpha = 1) # opacity
    panel.plot([pos, pos], [P5, P25], 
            marker = 'o',
            markerfacecolor = 'red',
            markeredgewidth = 0, # width of line around marker point, '0' for no marker width
            markersize = 0,
            color = 'black', # line color
            linewidth = 1, # line width, '0' for no line
            alpha = 1) # opacity
    
    ## Drawing horizontal lines in both ends to represent 5th and 95th percentile
    panel.plot([pos-(width/4), pos+(width/4)], [P95, P95], 
            marker = 'o',
            markerfacecolor = 'red',
            markeredgewidth = 0, # width of line around marker point, '0' for no marker width
            markersize = 0,
            color = 'black', # line color
            linewidth = 1, # line width, '0' for no line
            alpha = 1) # opacity
    panel.plot([pos-(width/4), pos+(width/4)], [P5, P5], 
            marker = 'o',
            markerfacecolor = 'red',
            markeredgewidth = 0, # width of line around marker point, '0' for no marker width
            markersize = 0,
            color = 'black', # line color
            linewidth = 1, # line width, '0' for no line
            alpha = 1) # opacity
    return


current_dir = os.getcwd()
memory_config_scripts_directory_path = os.path.join(current_dir, "comparative_plots", "memory_config_fraction")

with open(os.path.join(memory_config_scripts_directory_path, "accumulated_points_dict.json")) as infile:
    data_point_dict = json.load(infile)

fraction_wise_avg_list = [[[] for i in range(0, 800, 100)] for i in range(len(data_point_dict['points'][0]['profit']))]

for data_point in data_point_dict['points']:
    ue_count=data_point['UE_count']
    normalizing_base = data_point['dts_profit']
    for fraction_idx in range(len(data_point['profit'])):
        val=(data_point['profit'][fraction_idx]/normalizing_base)
        fraction_wise_avg_list[fraction_idx][(ue_count//100)].append(val)

for fraction_idx in range(len(fraction_wise_avg_list)):
    for ue_count_idx in range(1, len(fraction_wise_avg_list[fraction_idx])-1):
        # print(f"{ue_count_idx=}")
        fraction_wise_avg_list[fraction_idx][ue_count_idx+1]=np.array(fraction_wise_avg_list[fraction_idx][ue_count_idx+1])

figureWidth=13
figureHeight=10

plt.figure(figsize=(figureWidth, figureHeight))

panelWidth=11
panelHeight=7

panel1 = plt.axes([1/figureWidth, 1/figureHeight, panelWidth/figureWidth, panelHeight/figureHeight])

# color_pal = [(225/255,13/255,50/255),
# (242/255,50/255,54/255),
# (239/255,99/255,59/255),
# (244/255,138/255,30/255),
# (248/255,177/255,61/255),
# (143/255,138/255,86/255),
# (32/255,100/255,113/255),
# (42/255,88/255,132/255),
# (56/255,66/255,156/255),
# (84/255,60/255,135/255),
# (110/255,57/255,115/255),
# (155/255,42/255,90/255)
# ]

color_pal = ["#54bebe", "#76c8c8", "#98d1d1", "#badbdb", "#dedad2", "#e4bcad", "#df979e", "#d7658b", "#c80064"]

# color_pal = [
#     (248/255,174/255,51/255),
#     (88/255,85/255,120/255),
#     (60/255,62/255,100/255),
#     (192/255,41/255,46/255),
#     (230/255,87/255,43/255),
#     (81/255,116/255,95/255),
#     (120/255,172/255,145/255)
# ]
# color_pal = ["lightcyan", "deeppink", "darkviolet", "blue", "mistyrose", "green", "yellow"]
# print(f"{len(color_pal)=}, {len(fraction_wise_avg_list)=}")
# artisinal_boxplot(yvalues=yvalues, pos=10, width=0.5, color='white', panel=panel1)
xlims = [i for i in range(0, 71, 10)]
legend_elements = []
fraction_labels = [1, 5, 10, 15, 20, 25, 30]

done = False
for ue_count_idx_into_10 in xlims[1:-1]:
    for fraction_idx in range(len(fraction_wise_avg_list)):
        yvalues = fraction_wise_avg_list[fraction_idx][(ue_count_idx_into_10//10)]
        curr_color = color_pal[fraction_idx+1]
        curr_fraction_label = str(fraction_labels[fraction_idx]) + " " + ("bin" if (fraction_idx==0) else "bins")
        if fraction_idx == 2:
            median_linewidth = 2
            median_line_color = "blue"
        else:
            median_linewidth = 1
            median_line_color = "black"
        artisinal_boxplot(yvalues=yvalues, pos=ue_count_idx_into_10+fraction_idx, width=0.8, \
                        color=curr_color, panel=panel1, median_line_color=median_line_color, \
                        median_linewidth = median_linewidth)
        # panel1.boxplot(yvalues, positions=[i+j], widths=[0.5]) ### matplotlib custom function
        if not done:
            legend_elements.append(mplpatches.Patch(facecolor=curr_color, label=curr_fraction_label))
            # legend_elements.append(mplpatches.Patch(facecolor=curr_color, edgecolor='r',
            #              label=curr_fraction_label))
    done = True

for panel in [panel1]:
    panel.set_xlim(0,71)
    panel.set_ylim(0,3)

x_ticks_list = [i for i in range(3, 80, 10)]
panel1.set_xticks(x_ticks_list, [str(i) for i in range(0, 800, 100)], fontsize=15)
panel1.set_yticks([i/2 for i in range(7)], [str(i/2) for i in range(7)], fontsize=15)
panel.set_xlabel('No. of UEs in system', fontdict={'fontsize': 19})
panel.set_ylabel('Normalized Total System Profit', fontdict={'fontsize': 19})

panel1.tick_params(bottom=True, labelbottom=True, 
                   left=True, labelleft=True,
                   right=False, labelright=False,
                   top=False, labeltop=False
                   )
panel1.legend(handles=legend_elements, fontsize="15", ncol=3)

plt.savefig(os.path.join(memory_config_scripts_directory_path, "memory_config_fraction_boxplot.png"), dpi = 600) # Saves in the current directory
plt.savefig(os.path.join(current_dir, "plots", "memory_config_fraction_boxplot.png"), dpi = 600) # Saves in the current directory

# plt.show()
 






# fig = plt.figure(figsize =(10, 7))
 
# # Creating axes instance
# ax = fig.add_axes([0, 0, 1, 1])
 
# # Creating plot
# bp = ax.boxplot(fraction_wise_avg_list, patch_artist = True)
# # bp = plt.boxplot(fraction_wise_avg_list)

# # show plot
# plt.show()