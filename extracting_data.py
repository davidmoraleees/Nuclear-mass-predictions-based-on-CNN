import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_fwf(
    'mass_1.mas20.txt',
    usecols=(1, 2, 3, 4, 6, 9, 10, 11, 13, 16, 17, 21, 22),
    names=['N-Z', 'N', 'Z', 'A', 'Element', 'mass_exc',
           'mass_exc_unc', 'bind_ene', 'bind_ene_unc', 'beta_ene',
           'beta_ene_unc', 'atomic_mass', 'atomic_mass_unc'],
    widths=(1, 3, 5, 5, 5, 1, 3, 4, 1, 14, 12, 13, 1, 10, 1, 2, 13, 11, 1, 3, 1, 13, 12, 1),
    header=28,
    index_col=False
)

df = df.replace({'#': ''}, regex=True)
df['Z'] = df['Z'].astype(int)
df['A'] = df['A'].astype(int)
df['bind_ene'] = df['bind_ene'].astype(float)

# Theoretical model calculations
av = 15.8 #MeV
aS = 18.3
ac = 0.66
aA = 23.2
ap = 11.2

df['delta'] = np.where((df['Z'] % 2 == 0) & (df['N'] % 2 == 0), -1,  # Z even and N even
                np.where(df['A'] % 2 == 1, 0,  # A odd
                np.where((df['Z'] % 2 == 1) & (df['N'] % 2 == 1), 1, np.nan)))  # Z odd and N odd

df['delta'] = df['delta'].astype(int)

Z = df['Z']
N = df['N']
A = df['A']
delta = df['delta']

df = df[(df['N'] >= 8) & (df['N'] < 180) & (df['Z'] >= 8) & (df['Z'] < 120)] #Restrictions 

bind_ene_teo = (av*A-aS*A**(2/3)-ac*Z**2*A**(-1/3)-aA*(A-2*Z)**2/A-ap*delta*A**(-1/2))/A*1000 #keV

df['bind_ene_teo'] = bind_ene_teo
df['Diff_bind_ene'] = df['bind_ene'] - df['bind_ene_teo']

df.to_csv('data_cleaned.csv', sep=';', index=False)
print(df.head())

plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['N'], df['Z'], c=df['Diff_bind_ene'], cmap='jet', edgecolor='None', s=25)
cbar = plt.colorbar(scatter)
plt.xlabel('N')
plt.ylabel('Z') 
plt.grid()
plt.savefig('bind_teoexp_dif.png')
plt.show()

#3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')  
scatter = ax.scatter(df['Z'], df['N'], df['Diff_bind_ene'], c=df['Diff_bind_ene'], cmap='jet', edgecolor='None', s=25)
ax.set_xlabel('Z')
ax.set_ylabel('N')
cbar = plt.colorbar(scatter, ax=ax)
plt.savefig('bind_teoexp_dif_3D.png') 
plt.show()




