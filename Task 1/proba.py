import matplotlib.pyplot as plt
import pandas as pd

header = ['X','Y']
data = pd.read_csv('ships.csv', names=header, header = 1)

x = data['X']
y = data['Y']

bla = -24.6567 + 122.60 * x
bla = 250 ** x
print(bla)
plt.figure(1)
plt.subplot(211)
plt.plot(x,y,'*')
plt.plot(x,bla)
plt.show()