import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

df = pd.read_pickle('./Final_Prediction.csv')

explosions = df[df['y_perd'] == 'single_exposion']
print("We got {} explosions.".format(len(explosions)))

frame_index, width_index, height_index = explosions['frame_index'],
explosions['width_index'],
explosions['height_index']

fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection='3d')
ax.scatter3D(frame_index, width_index, height_index)
plt.title('Explosion Map')
plt.xlabel('Frame')
plt.ylabel('Width')
plt.set_zlabel('Height')
plt.show()
