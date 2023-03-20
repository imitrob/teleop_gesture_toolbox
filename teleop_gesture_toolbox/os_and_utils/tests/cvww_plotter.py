import numpy as np
import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(7,5))

# set height of bar
teleop =   [48, 118, 119]

gesture1 = [35, 98, 104]
gesture1_ = [16,43,56]

gesture2 = [34, 40, 39]
gesture2_ = [7, 9, 8]

# Set position of bar on X axis
br1 = np.arange(len(teleop))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
plt.grid(axis='y')
# Make the plot
plt.bar(br1, teleop, color ='r', width = barWidth,
        edgecolor ='grey', label ='Tele-operation')


plt.bar(br2, gesture1, color ='lightgreen', width = barWidth,
        edgecolor ='grey', label ='Low-lvl ActGs \nRunning')
plt.bar(br2, gesture1_, color ='g', width = barWidth,
        edgecolor ='grey', label ='Low-lvl ActGs \nUser Performs')

plt.bar(br3, gesture2, color ='lightblue', width = barWidth,
        edgecolor ='grey', label ='Hi-lvl ActGs \nRunning')
plt.bar(br3, gesture2_, color ='b', width = barWidth,
        edgecolor ='grey', label ='Hi-lvl ActGs \nUser Performs')


# Adding Xticks
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.xlabel('Scenario', fontsize = 15)
plt.ylabel('Time to complete [s]', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(teleop))],
        ['Put to bowl', 'Swap objects', 'Put to occupied bowl'])


plt.legend()
plt.savefig("/home/petr/Pictures/CVWW22_plot_taskcompletion_font.eps")
plt.show()
