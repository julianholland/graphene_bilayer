from pdb import run
from ase.build import graphene_nanoribbon, graphene, molecule, rotate
from ase.build.supercells import make_supercell
import numpy as np
from ase.visualize import view
from ase import Atoms, Atom
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.io import write, Trajectory
import matplotlib.pyplot as plt
from tqdm import tqdm
from ase.io.trajectory import TrajectoryWriter

# gradients_for_zig_zag
# y=0x, y=0.57x

# gradients for armchair
# y=infinty*x 

#make square
def make_square_graphene(size=20):
    layer=graphene()
    layer.positions=np.array([1.4,0,0])+layer.positions
    p=np.array([[1,-1,0],[2,2,0],[0,0,1]])
    orthogonal_structure=make_supercell(layer,p)
    orthogonal_structure.rotate(30, 'z', rotate_cell=True)
    over_size=size*3
    new_p=np.array([[over_size,0,0],[0,over_size,0],[0,0,1]])
    large_orthogonal_struc=make_supercell(orthogonal_structure,new_p)
    # large_orthogonal_struc.rotate(-4.49197,'z',center=[large_orthogonal_struc.cell[0,0],large_orthogonal_struc.cell[1,1], 0])
    cell_unit=large_orthogonal_struc.cell[0,0]*(1/3)
    mask=(large_orthogonal_struc.positions[:,0]<=2*cell_unit) & (large_orthogonal_struc.positions[:,1]<=2*cell_unit) & (large_orthogonal_struc.positions[:,0]>=cell_unit) & (large_orthogonal_struc.positions[:,1]>=cell_unit)
    square_struc=large_orthogonal_struc[mask]
    square_struc.positions=square_struc.positions-np.array([cell_unit,cell_unit,0])
    square_struc.cell=[cell_unit, cell_unit,0]
    # view(square_struc)
    return square_struc
    
def make_graphene_hexagon(square_struc):
    cut_grad=0.57719
    # cut_grad=0.477
    nudge=0
    mask=[]
    unit_cell_length=square_struc.cell[1,1]
    a=unit_cell_length*0.28
    for atom in square_struc:
        # side lengths should be 0.625 of the square to be even 
        if atom.position[1]+nudge>atom.position[0]*cut_grad+(unit_cell_length-a):
        # if atom.position[1]>cut_grad*atom.position[0]+(unit_cell_length-a):
            # print(atom, 'True')
            mask.append(False)
        elif atom.position[1]+nudge<-atom.position[0]*cut_grad+(a):
        # elif atom.position[1]<-cut_grad*atom.position[0]+a:
            mask.append(False)
        elif atom.position[1]+nudge>-atom.position[0]*(cut_grad)+(unit_cell_length+a):
        # elif atom.position[1]>-cut_grad*atom.position[0]+unit_cell_length+a:
            mask.append(False)
        elif atom.position[1]+nudge<atom.position[0]*(cut_grad)-a:
        # elif atom.position[1]<cut_grad*atom.position[0]-a:
            mask.append(False)
        else:
            mask.append(True)
    rough_hexagonal_struc=square_struc[mask]
    rough_hexagonal_struc.cell=square_struc.cell*2
    # view(rough_hexagonal_struc)
    return rough_hexagonal_struc

def trim_dangling_carbons(rough_hexagonal_struc):
    mask=[]
    # trim C atoms with < 1 bond
    for i in range(len(rough_hexagonal_struc)):
        distances=rough_hexagonal_struc.get_distances(i, indices=list(range(len(rough_hexagonal_struc))) ,mic=True)
        distance_mask=distances<1.43
        bonds=len(distances[distance_mask])-1
        # print('bonds',bonds)
        if bonds == 1:
            mask.append(False)
        else:
            mask.append(True)
    
    hexagon_struc=rough_hexagonal_struc[mask]
    hexagon_struc_no_cell=Atoms(hexagon_struc.symbols, hexagon_struc.positions)
    return hexagon_struc_no_cell

def terminate_graphene(hexagon_struc, terminator, distance=1):

    # create copy of structures to edit

    teriminated_hexagon_struc=hexagon_struc.copy()

    # add termination
    evaluator_pos_list=[]
    for i in range(len(hexagon_struc)):
        # find number of bonds for each atoms
        distances=hexagon_struc.get_distances(i, indices=list(range(len(hexagon_struc))) ,mic=True)
        distance_mask=(distances<1.43) & (distances>0.1)
        neighbour_atoms=hexagon_struc[distance_mask]
        bonds=len(neighbour_atoms)
        
        # select atoms on edge (i.e. those with only two bonds)
        if bonds == 2:
            # determine direction the termination needs to face based on neighbor atoms
            evaluator_pos=np.round(hexagon_struc[i].position-(neighbour_atoms[0].position+neighbour_atoms[1].position)/2, decimals=5)
            evaluator_sign=np.sign(evaluator_pos)

            # scale the direction for diagnonals and chosen distance
            scaled_evaluator_sign=(evaluator_sign*distance)/np.sqrt(np.sum(np.abs(evaluator_sign**2)))

            # Add terminator at scaled positon to main structure
            terminator.position=hexagon_struc[i].position+scaled_evaluator_sign
            #     terminator.positions=terminator.positions+hexagon_struc[i].position+scaled_evaluator_sign
            teriminated_hexagon_struc+=terminator
    
    return teriminated_hexagon_struc

def make_layers(structure, layers=2, angle=0, interlayer_distance=3.355):
    height=interlayer_distance
    layered_structure=structure.copy()
    center=layered_structure.get_center_of_mass()
    individual_layer_copy=structure.copy()
    for layer in range(layers-1):
        individual_layer_copy.positions=individual_layer_copy.positions+[0,0,height]
        individual_layer_copy.rotate(angle, 'z', center)
        layered_structure+=individual_layer_copy
    
    return layered_structure

def make_cell(layered_structure, vacuum=10):
    layered_structure.center(vacuum=vacuum)
    return layered_structure
    

def run_all(size, terminator=Atom('H'), terminator_bond_length=1.09, layers=2, angle=0, interlayer_distance=3.355, add_cell=True, cell_vacuum=10, verbose=1):
    square_graphene=make_square_graphene(size)
    hexagonal_graphene=make_graphene_hexagon(square_graphene)
    neat_hexagonal_graphene=trim_dangling_carbons(hexagonal_graphene)
    teriminated_structure=terminate_graphene(neat_hexagonal_graphene, terminator, terminator_bond_length)
    layered_structure=make_layers(teriminated_structure, layers, angle, interlayer_distance)
    if add_cell:
        layered_structure=make_cell(layered_structure, cell_vacuum)
    # view(layered_structure)
    if verbose > 0:
        print('atom number:', len(layered_structure), f'({layered_structure.get_chemical_formula()})')
    return layered_structure

def unique_environment_analysis(size, write_trajectory=False):
    initial_structure=run_all(size, angle=0, add_cell=False, verbose=1)
    chemical_environements_list=[]
    twist_trajectory=[]
    for angle in tqdm(range(0,361)):
        twisted_bilayer=run_all(size, angle=angle/2, add_cell=False, verbose=0)
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


oli=Atoms(symbols=['Li','O'],positions=[[1.96,0,0],[0,0,0]])
h=Atom('H')

# for i in range(16,21):
#     a=run_all(i)
#     view(a)

twist_data_list=[]
twist_stat_list=[]
for size in range(20,21):
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



