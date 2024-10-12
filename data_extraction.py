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
    df['bind_ene_teo_total'] = df['bind_ene_teo']*df['A']
    df['Diff_bind_ene'] = df['bind_ene'] - df['bind_ene_teo']

    df = df[(df['N'] >= 8) & (df['N'] < 180) & (df['Z'] >= 8) & (df['Z'] < 120)] # Restrictions for our study case

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

rmse_2016 = np.sqrt(np.mean(df2016['Diff_bind_ene'] ** 2))
print('RMSE liquid droplet model (2016): ', rmse_2016)
rmse_2020 = np.sqrt(np.mean(df2020['Diff_bind_ene'] ** 2))
print('RMSE liquid droplet model (2020): ', rmse_2020)


#Plots of AME2020 dataset:
#Plot of the theoretical binding energy as a function of Z and N
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df2020['N'], df2020['Z'], c=df2020['bind_ene_teo'], cmap='jet', edgecolor='None',
                      s=25, vmin=df2020['bind_ene_teo'].min(), vmax=df2020['bind_ene_teo'].max())
cbar = plt.colorbar(scatter)
cbar.set_label('(MeV)')
magic_numbers = [8, 20, 28, 50, 82, 126]
for magic in magic_numbers:
    plt.axvline(x=magic, color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(y=magic, color='gray', linestyle='--', linewidth=0.5)
plt.xticks(magic_numbers)
plt.yticks(magic_numbers)
plt.xlabel('N')
plt.ylabel('Z') 
plt.title('Theoretical binding energy AME2020')
plt.savefig('Binding energy plots/bind_teo.png')
plt.show()

#Plot of the experimental binding energy as a function of Z and N
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df2020['N'], df2020['Z'], c=df2020['bind_ene'], cmap='jet', edgecolor='None',
                      s=25, vmin=df2020['bind_ene_teo'].min(), vmax=df2020['bind_ene_teo'].max())
cbar = plt.colorbar(scatter)
cbar.set_label('(MeV)')
magic_numbers = [8, 20, 28, 50, 82, 126]
for magic in magic_numbers:
    plt.axvline(x=magic, color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(y=magic, color='gray', linestyle='--', linewidth=0.5)
plt.xticks(magic_numbers)
plt.yticks(magic_numbers)
plt.xlabel('N')
plt.ylabel('Z') 
plt.title('Experimental binding energy AME2020')
plt.savefig('Binding energy plots/bind_exp.png')
plt.show()

#Plot of the difference between theoretical and experimental binding energies
plt.figure(figsize=(10, 6))
norm = TwoSlopeNorm(vmin=df2020['Diff_bind_ene'].min(), vcenter=0, vmax=df2020['Diff_bind_ene'].max())
scatter = plt.scatter(df2020['N'], df2020['Z'], c=df2020['Diff_bind_ene'],
                      cmap='seismic', norm=norm, edgecolor='None', s=25)
cbar = plt.colorbar(scatter)
cbar.set_label('(MeV)')
magic_numbers = [8, 20, 28, 50, 82, 126]
for magic in magic_numbers:
    plt.axvline(x=magic, color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(y=magic, color='gray', linestyle='--', linewidth=0.5)
plt.xticks(magic_numbers)
plt.yticks(magic_numbers)
plt.xlabel('N')
plt.ylabel('Z') 
plt.title('Difference exp-teo binding energy AME2020 ')
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
plt.title('3D Difference exp-teo binding energy AME2020 ')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('(MeV)')
plt.savefig('Binding energy plots/bind_teoexp_dif_3D.png') 
plt.show()


#Experimental nuclear shell gaps (\Delta_{2n} and \Delta_{2p})
def calculate_shell_gaps(df, element, axis, type, column):
    #Neutrons--> element = n, axis = Z; Protons--> element = p, axis = N
    #type = exp or teo; column = 'bind_ene_total' or 'bind_ene_teo_total'
    df[f'bind_ene_{element}+2_{type}'] = df.groupby(axis)[column].shift(-2)
    df[f'bind_ene_{element}-2_{type}'] = df.groupby(axis)[column].shift(2)
    df[f'delta_2{element}_{type}'] = df[f'bind_ene_{element}-2_{type}'] - 2 * df[column] + df[f'bind_ene_{element}+2_{type}']
    return df

def plot_shell_gaps(df, gap_col, type, title, filename, vmin, vmax, xlim, ylim): #gap_col=delta_2n or delta_2p
    plot_name = "{}_{}".format(gap_col, type)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['N'], df['Z'], c=df[plot_name]*(-1), cmap='jet', edgecolor='None', s=25, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(scatter)
    cbar.set_label('(MeV)')

    magic_numbers = [8, 20, 28, 50, 82, 126]
    for magic in magic_numbers:
        plt.axvline(x=magic, color='gray', linestyle='--', linewidth=0.5)
        plt.axhline(y=magic, color='gray', linestyle='--', linewidth=0.5)

    plt.xticks(magic_numbers)
    plt.yticks(magic_numbers)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('N')
    plt.ylabel('Z')
    plt.title("{} {}".format(title, type))
    plt.savefig(filename)
    plt.show()

df2020 = calculate_shell_gaps(df2020, 'n', 'Z', 'exp', 'bind_ene_total') 
df2020 = calculate_shell_gaps(df2020, 'p', 'N', 'exp', 'bind_ene_total') 

min_value = min(df2020['delta_2n_exp'].min(), df2020['delta_2p_exp'].min()) * (-1) #Same colorbar range for both plots
max_value = max(df2020['delta_2n_exp'].max(), df2020['delta_2p_exp'].max()) * (-1)
xlim = (min(df2020['N'].min(), 0), max(df2020['N'].max() + 10, 0)) #Same limits for both plots
ylim = (0, df2020['Z'].max() + 10)  

plot_shell_gaps(df2020, 'delta_2n', 'exp', 'Neutron shell gaps', 'Binding energy plots/neutron_shell_gaps_exp.png',
                min_value, max_value, xlim=xlim, ylim=ylim)
plot_shell_gaps(df2020, 'delta_2p', 'exp', 'Proton shell gaps', 'Binding energy plots/proton_shell_gaps_exp.png',
                min_value, max_value, xlim=xlim, ylim=ylim)


df2020 = calculate_shell_gaps(df2020, 'n', 'Z', 'teo', 'bind_ene_teo_total') 
df2020 = calculate_shell_gaps(df2020, 'p', 'N', 'teo', 'bind_ene_teo_total') 

min_value = min(df2020['delta_2n_teo'].min(), df2020['delta_2p_teo'].min()) * (-1) #Same colorbar range for both plots
max_value = max(df2020['delta_2n_teo'].max(), df2020['delta_2p_teo'].max()) * (-1)
xlim = (min(df2020['N'].min(), 0), max(df2020['N'].max() + 10, 0)) #Same limits for both plots
ylim = (0, df2020['Z'].max() + 10)  

plot_shell_gaps(df2020, 'delta_2n', 'teo', 'Neutron shell gaps', 'Binding energy plots/neutron_shell_gaps_teo.png',
                min_value, max_value, xlim=xlim, ylim=ylim)
plot_shell_gaps(df2020, 'delta_2p', 'teo', 'Proton shell gaps', 'Binding energy plots/proton_shell_gaps_teo.png',
                min_value, max_value, xlim=xlim, ylim=ylim)





