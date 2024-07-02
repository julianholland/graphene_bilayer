from twist_analysis import unique_environment_analysis
from bilayer_maker import run_all
from ase.atom import Atom
from ase.atoms import Atoms
from ase.visualize import view
import numpy as np

oli=Atoms(symbols=['Li','O'],positions=[[1.96,0,0],[0,0,0]])
h=Atom('H')

# for i in range(11,21):
#     a=run_all(i, angle=33)
#     view(a)

twist_data_list=[]
twist_stat_list=[]
for size in range(1,26):
    twist_data, structure=unique_environment_analysis(size, write_trajectory=True)
    twist_data=np.array(twist_data)
    twist_data_list.append(twist_data)
    filename=str(size)+'_twist_data.csv'
    max_chemical_envs=np.max(twist_data[:,1])
    best_angles=np.where(twist_data[:,1]==max_chemical_envs)
    number_of_atoms=len(structure[np.array(structure.get_chemical_symbols())=='C'])
    twist_stat=[max_chemical_envs, number_of_atoms, best_angles[0][0]]
    twist_stat_list.append(twist_stat)
    np.savetxt(filename, twist_data, delimiter=',')
    # plt.plot(twist_data[:,1], 'b')
    # plt.show(block=False)
twist_stat_array=np.array(twist_stat_list)
print(twist_stat_array)
np.savetxt('twist_stat.csv', twist_stat_array, delimiter=',')