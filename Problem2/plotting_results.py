#  Yamini Ananth
# COMS 6998: High Performance ML
# PLOTTING PART B

import matplotlib.pyplot as plt

K_values = [1, 5, 10, 50, 100]
CPU = [0.000536,0.003221,0.007444,0.045066, 0.090538]

Q2_SITUATION1 = [0.066918,0.298293,0.593383,2.962030,5.941754]
Q2_SITUATION2 = [0.001494,0.007408,0.014864,0.073399,0.142169]
Q2_SITUATION3 = [0.000025,0.000085,0.014864,0.000736,0.001457]

Q3_SITUATION1 = [0.063650,0.298010,0.592224,2.961198, 5.922734]
Q3_SITUATION2 = [0.001515,0.007442,0.014961,0.071310,0.137678]
Q3_SITUATION3 = [0.000041,0.000105,0.000177,0.000762,0.001485]


# Plotting Q2
plt.scatter(K_values, CPU, label='Host/CPU')
plt.scatter(K_values, Q2_SITUATION1, label='1 block/1 thread')
plt.scatter(K_values, Q2_SITUATION2, label='1 block/256 threads')
plt.scatter(K_values, Q2_SITUATION3, label='Any blocks/256 threads')
plt.legend()

plt.xlabel('K')
plt.ylabel('Time (sec)')
plt.title('Q2: No Unified Memory')
plt.yscale('log')
plt.savefig('Q2.png')

# Clearing the plot
plt.cla()

# Plotting Q3
plt.scatter(K_values, CPU, label='Host/CPU')
plt.scatter(K_values, Q3_SITUATION1, label='1 block/1 thread')
plt.scatter(K_values, Q3_SITUATION2, label='1 block/256 threads')
plt.scatter(K_values, Q3_SITUATION3, label='Any blocks/256 threads')
plt.legend()
plt.xlabel('K')
plt.ylabel('Time (sec)')
plt.title('Q3: Unified Memory')
plt.yscale('log')
plt.savefig('Q3.png')