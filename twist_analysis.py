from bilayer_maker import run_all
from tqdm import tqdm
import numpy as np
from ase.io.trajectory import TrajectoryWriter

def unique_environment_analysis(size, write_trajectory=False, high_symmetry=False):
    initial_structure=run_all(size, angle=0, add_cell=True, verbose=1)
    chemical_environements_list=[]
    twist_trajectory=[]
    for angle in tqdm(range(0,361)):
        twisted_bilayer=run_all(size, angle=angle/2, add_cell=False, verbose=0, high_symmetry=high_symmetry)
        just_c=twisted_bilayer[np.array(twisted_bilayer.get_chemical_symbols())=='C']
        all_distances=just_c.get_all_distances()
        # print(np.size(all_distances))
        # print(int(len(just_c)/2))
        interlayer_distances=all_distances[int(len(just_c)/2):, :int(len(just_c)/2)]
        min_dist_list=np.min(interlayer_distances, axis=1)
        # print(min_dist_list)


        # layer_1=just_c[just_c.positions[:,2]<1]
        # layer_2=just_c[just_c.positions[:,2]>1]
        # min_dist_list=[]
        # for lower_atom_index in range(len(layer_1)):
        #     dist_list=[]
        #     for upper_atom_index in range(len(layer_2)):
        #         dist=np.linalg.norm(layer_1[lower_atom_index].position-layer_2[upper_atom_index].position)
        #         dist_list.append(dist)
        #     min_dist_list.append(np.min(dist_list))
            
        
        chemical_environements=[angle,len(np.unique(np.round(min_dist_list,decimals=5)))]
        chemical_environements_list.append(chemical_environements)
        if write_trajectory:
            twist_trajectory.append(twisted_bilayer)
    
    if write_trajectory:
        writer=TrajectoryWriter(str(size)+'_'+str(twisted_bilayer.get_chemical_formula())+'_bilayer_rotaion.traj')
        for structure in twist_trajectory:
            writer.write(structure)

    return chemical_environements_list, initial_structure