import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Nombre del archivo original y del archivo nuevo
input_file = 'AME2016.txt'
output_file='AME2016_modified.txt'

# Abrir el archivo original para leer
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line_number, line in enumerate(infile, start=1):
        # Si la línea es menor que 40, se escribe sin cambios
        if line_number < 40:
            outfile.write(line)
        else:
            # A partir de la línea 40, modificar la primera columna
            modified_line = line.lstrip()  # Eliminar espacios al inicio de la línea
            
            # Reemplazar el primer '0' encontrado al inicio de la línea por un espacio vacío
            if modified_line.startswith('0 '):  # Verifica si empieza con '0 '
                modified_line = modified_line.replace('0 ', ' ', 1)  # Reemplaza solo el primer '0 '

            outfile.write(modified_line)  # Escribir la línea modificada

print("Ceros en la primera columna reemplazados por espacios a partir de la línea 40.")



with open('AME2016_modified.txt', 'r') as file:
    header=['N-Z', 'N', 'Z', 'A', 'Element', 'O', 'mass_exc',
           'mass_exc_unc', 'bind_ene', 'bind_ene_unc', 'beta', 'beta_ene',
           'beta_ene_unc', 'A2', 'atomic_mass', 'atomic_mass_unc']
    for i in range(39):
        next(file) #Data
    data = [line.strip() for line in file]

data_rows = [line.split() for line in data]
df = pd.DataFrame(data_rows, columns=header)  

#columns_to_drop = ['index', 'O', 'beta', 'A2']
#df.drop(columns=columns_to_drop, inplace=False)





df.to_csv('AME2016_cleaned.csv', index=False, header=True, sep=';')



print(df.head())