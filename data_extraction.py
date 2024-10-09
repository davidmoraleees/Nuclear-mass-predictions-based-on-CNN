import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import TwoSlopeNorm


# Extracting data from WS4 file
with open('data/WS4.txt', 'r') as file:
    for i in range(29):
        if i == 27:  
            header = file.readline().strip().split() 
        else:
            next(file) 
    data = [line.strip() for line in file]

data_rows = [line.split() for line in data]
dfWS4= pd.DataFrame(data_rows, columns=header)  
dfWS4.to_csv('data/WS4_cleaned.csv', index=False, header=True, sep=';')
print("WS4 dataset: \n", dfWS4.head(), "\n")


# Function to extract data from AME2020 and AME2016 files
def process_file(filename, header, widths, columns, column_names, year):
    df = pd.read_fwf(
        filename,
        usecols=columns,
        names=column_names,
        widths=widths,
        header=header,
        index_col=False
    )
    df = df.replace({'#': ''}, regex=True) 
    df['Z'] = df['Z'].astype(int) 
    df['A'] = df['A'].astype(int) 
    df['bind_ene'] = df['bind_ene'].astype(float)/1000
    df['bind_ene_total'] = df['bind_ene']*df['A']

    df['delta'] = np.where((df['Z'] % 2 == 0) & (df['N'] % 2 == 0), -1,  # Z and N even
                   np.where(df['A'] % 2 == 1, 0,  # A odd
                   np.where((df['Z'] % 2 == 1) & (df['N'] % 2 == 1), 1, np.nan)))  # Z and N odds
    df['delta'] = df['delta'].astype(int)

    df = df[(df['N'] >= 8) & (df['N'] < 180) & (df['Z'] >= 8) & (df['Z'] < 120)] # Restrictions for our study case

    # Theoretical model
    av = 15.8  # MeV
    aS = 18.3
    ac = 0.66
    aA = 23.2
    ap = 11.2

    A = df['A']
    Z = df['Z']
    delta = df['delta']

    bind_ene_teo = (av*A-aS*A**(2/3)-ac*Z**2*A**(-1/3)-aA*(A-2*Z)**2/A-ap*delta*A**(-1/2))/A  # MeV
    df['bind_ene_teo'] = bind_ene_teo
    df['Diff_bind_ene'] = df['bind_ene'] - df['bind_ene_teo']

    df.to_csv(f'data/mass{year}_cleaned.csv', sep=';', index=False)
    return df

columns_2020 = (1, 2, 3, 4, 6, 9, 10, 11, 13, 16, 17, 21, 22)
widths_2020 = (1, 3, 5, 5, 5, 1, 3, 4, 1, 14, 12, 13, 1, 10, 1, 2, 13, 11, 1, 3, 1, 13, 12, 1)
column_names_2020 = ['N-Z', 'N', 'Z', 'A', 'Element', 'mass_exc', 'mass_exc_unc', 'bind_ene', 'bind_ene_unc',
                     'beta_ene', 'beta_ene_unc', 'atomic_mass', 'atomic_mass_unc']
header_2020 = 28

columns_2016 = (1, 2, 3, 4, 6, 9, 10, 11, 12, 15, 16, 18, 20, 21)
widths_2016 = [1, 3, 5, 5, 5, 1, 3, 4, 1, 13, 11, 11, 9, 1, 2, 11, 9, 1, 3, 1, 12, 11, 5]
column_names_2016 = ['index', 'N-Z', 'N', 'Z', 'A', 'empty', 'Element', 'empty2', 'empty3', 'mass_exc', 
                     'mass_exc_unc', 'bind_ene', 'bind_ene_unc', 'empty4', 'empty5', 'beta_ene', 'beta_ene_unc',
                     'empty6', 'A2', 'empty7', 'atomic_mass', 'atomic_mass_unc']
header_2016 = 31

df2020 = process_file('data/mass2020.txt', header_2020, widths_2020, columns_2020, column_names_2020, 2020)
print("AME2020 dataset: \n", df2020.head(), "\n")

df2016 = process_file('data/mass2016.txt', header_2016, widths_2016, columns_2016, column_names_2016, 2016)
print("AME2016 dataset: \n", df2016.head(), "\n")


#Plots of AME2020 dataset:
#Plot of the theoretical binding energy as a function of Z and N
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df2020['N'], df2020['Z'], c=df2020['bind_ene_teo'], cmap='jet', edgecolor='None',
                      s=25, vmin=df2020['bind_ene_teo'].min(), vmax=df2020['bind_ene_teo'].max())
cbar = plt.colorbar(scatter)
cbar.set_label('(MeV)')
plt.xlabel('N')
plt.ylabel('Z') 
plt.title('Theoretical binding energy AME2020')
plt.grid()
plt.savefig('Binding energy plots/bind_teo.png')
plt.show()

#Plot of the experimental binding energy as a function of Z and N
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df2020['N'], df2020['Z'], c=df2020['bind_ene'], cmap='jet', edgecolor='None',
                      s=25, vmin=df2020['bind_ene_teo'].min(), vmax=df2020['bind_ene_teo'].max())
cbar = plt.colorbar(scatter)
cbar.set_label('(MeV)')
plt.xlabel('N')
plt.ylabel('Z') 
plt.title('Experimental binding energy AME2020')
plt.grid()
plt.savefig('Binding energy plots/bind_exp.png')
plt.show()

#Plot of the difference between theoretical and experimental binding energies
plt.figure(figsize=(10, 6))
norm = TwoSlopeNorm(vmin=df2020['Diff_bind_ene'].min(), vcenter=0, vmax=df2020['Diff_bind_ene'].max())
scatter = plt.scatter(df2020['N'], df2020['Z'], c=df2020['Diff_bind_ene'],
                      cmap='seismic', norm=norm, edgecolor='None', s=25)
cbar = plt.colorbar(scatter)
cbar.set_label('(MeV)')
plt.xlabel('N')
plt.ylabel('Z') 
plt.title('Difference exp-teo binding energy AME2020 ')
plt.grid()
plt.savefig('Binding energy plots/bind_teoexp_dif.png')
plt.show()

#3D plot of the difference between theoretical and experimental binding energies
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')  
norm = TwoSlopeNorm(vmin=df2020['Diff_bind_ene'].min(), vcenter=0, vmax=df2020['Diff_bind_ene'].max())
scatter = ax.scatter(df2020['Z'], df2020['N'], df2020['Diff_bind_ene'], c=df2020['Diff_bind_ene'],
                     cmap='seismic', norm=norm, edgecolor='None', s=25)
ax.set_xlabel('Z')
ax.set_ylabel('N')
plt.title('3D difference exp-teo binding energy AME2020 ')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('(MeV)')
plt.savefig('Binding energy plots/bind_teoexp_dif_3D.png') 
plt.show()


#Experimental nuclear shell gaps (\Delta_{2n} and \Delta_{2p})
def calculate_shell_gaps(df, element, axis): #Neutrons--> element=n, axis=Z; Protons--> element=p, axis=N
    df[f'bind_ene_{element}+2'] = df.groupby(axis)['bind_ene_total'].shift(-2)
    df[f'bind_ene_{element}-2'] = df.groupby(axis)['bind_ene_total'].shift(2)
    df[f'delta_2{element}'] = df[f'bind_ene_{element}-2'] - 2 * df['bind_ene_total'] + df[f'bind_ene_{element}+2']
    return df

def plot_shell_gaps(df, gap_col, title, filename, vmin, vmax, xlim=None, ylim=None): #gap_col=delta_2n or delta_2p
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['N'], df['Z'], c=df[gap_col]*(-1), cmap='jet', edgecolor='None', s=15, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(scatter)
    cbar.set_label('(MeV)')

    magic_numbers = [8, 20, 28, 50, 82, 126]
    for magic in magic_numbers:
        plt.axvline(x=magic, color='gray', linestyle='--', linewidth=0.5)
        plt.axhline(y=magic, color='gray', linestyle='--', linewidth=0.5)
        plt.text(magic + 0.5, 0, str(magic), color='black', fontsize=10, ha='center', va='bottom')
        plt.text(0, magic - 0.5, str(magic), color='black', fontsize=10, ha='left', va='center')

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.xlabel('N')
    plt.ylabel('Z')
    plt.title(title)
    plt.savefig(filename)
    plt.show()

df2020 = calculate_shell_gaps(df2020, 'n', 'Z') 
df2020 = calculate_shell_gaps(df2020, 'p', 'N') 

min_value = min(df2020['delta_2n'].min(), df2020['delta_2p'].min()) * (-1) #Same colorbar range for both plots
max_value = max(df2020['delta_2n'].max(), df2020['delta_2p'].max()) * (-1)
xlim = (min(df2020['N'].min(), 0), max(df2020['N'].max() + 10, 0)) #Same limits for both plots
ylim = (0, df2020['Z'].max() + 10)  

plot_shell_gaps(df2020, 'delta_2n', 'Neutron shell gaps', 'Binding energy plots/neutron_shell_gaps.png',
                min_value, max_value, xlim=xlim, ylim=ylim)
plot_shell_gaps(df2020, 'delta_2p', 'Proton shell gaps', 'Binding energy plots/proton_shell_gaps.png',
                min_value, max_value, xlim=xlim, ylim=ylim)

