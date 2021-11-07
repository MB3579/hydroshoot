"""This is an example on running HydroShoot on a potted grapevine with a
simple shoot architecture.
"""

from os import getcwd
import pdb

from openalea.mtg import traversal
from openalea.plantgl.all import Scene
from hydroshoot import architecture, display, model

# =============================================================================
# Construct the plant mock-up
# =============================================================================

# Path for plant digitalization data.
choose_scenario = 2

if choose_scenario == 1:
    g,scene = architecture.mtg_load(str(getcwd()) + '/','10_days_sim')
    #pdb.set_trace()

else:
    g = architecture.vine_mtg('digit.input')

    # Local Coordinates Correction
    for v in traversal.iter_mtg2(g, g.root):
        architecture.vine_phyto_modular(g, v)
        architecture.vine_axeII(g, v, pruning_type='avg_field_model', N_max=6,
                            insert_angle=90, N_max_order=6)
        architecture.vine_petiole(g, v, pet_ins=90., pet_ins_cv=0.,
                              phyllo_angle=180.)
        architecture.vine_leaf(g, v, leaf_inc=-45., leaf_inc_cv=100.,
                           lim_max=12.5, lim_min=5., order_lim_max=5.5,
                           max_order=55, rand_rot_angle=90.,
                           cordon_vector=None)
        architecture.vine_mtg_properties(g, v)
        architecture.vine_mtg_geometry(g, v)
        architecture.vine_transform(g, v)

	scene = display.visu(g, def_elmnt_color_dict=True, scene=Scene(),
                     view_result=False)

# =============================================================================
# Run HydroShoot
# =============================================================================

#for ii in range(1,6):
#	model.run(g, str(getcwd()) + '/', scene, psi_soil = -0.5, gdd_since_budbreak = 760., param_index = ii)

ii = 1
model.run(g, str(getcwd()) + '/', scene, psi_soil = -0.5, gdd_since_budbreak = 760., param_index = ii)


