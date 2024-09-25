import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Càlculs amb el model teòric
av = 15.8 #MeV
aS = 18.3
ac = 0.66
aA = 23.2
ap = 11.2

df['delta'] = np.where((df['Z'] % 2 == 0) & (df['N'] % 2 == 0), -1,  # Z parell i N parell
                np.where(df['A'] % 2 == 1, 0,  # A senar
                np.where((df['Z'] % 2 == 1) & (df['N'] % 2 == 1), 1, np.nan)))  # Z senar i N senar

df['delta'] = df['delta'].astype(int)

Z = df['Z']
N = df['N']
A = df['A']
delta = df['delta']

bind_ene_teo = (av*A-aS*A**(2/3)-ac*Z**2*A**(-1/3)-aA*(A-2*Z)**2/A-ap*delta*A**(-1/2))/A*1000 #keV

df['bind_ene_teo'] = bind_ene_teo
df['Diferencia_bind_ene'] = df['bind_ene'] - df['bind_ene_teo']

df = df[(df['Diferencia_bind_ene'] <= 500) & (df['Diferencia_bind_ene'] >= -500)]

df.to_csv('data_cleaned.csv', sep=';', index=False)
print(df.head())


plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['N'], df['Z'], c=df['Diferencia_bind_ene'], cmap='jet', edgecolor='None', s=7)

cbar = plt.colorbar(scatter)

plt.xlabel('N')
plt.ylabel('Z')

plt.grid()
plt.savefig('bind_teoexp_dif.png')
plt.show()


