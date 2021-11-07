# -*- coding: utf-8 -*-
"""This module performs a complete comutation scheme: irradiance absorption, gas-exchange, hydraulic structure,
energy-exchange, and soil water depletion, for each given time step.
"""
import numpy as np
import pdb
from copy import deepcopy
from os.path import isfile
from datetime import datetime, timedelta
from pandas import read_csv, DataFrame, date_range, DatetimeIndex, merge

import openalea.mtg.traversal as traversal
from openalea.plantgl.all import Scene, surface

from hydroshoot import (architecture, irradiance, exchange, hydraulic, energy,
                        display, solver)
from hydroshoot.params import Params


def run(g, wd, scene=None, write_result=True, **kwargs):
    """
    Calculates leaf gas and energy exchange in addition to the hydraulic structure of an individual plant.

    :Parameters:
    - **g**: a multiscale tree graph object
    - **wd**: string, working directory
    - **scene**: PlantGl scene
    - **kwargs** can include:
        - **psi_soil**: [MPa] predawn soil water potential
	- **initial_psi_soil**: [MPa] predawn soil WP at the first timestep   
        - **gdd_since_budbreak**: [°Cd] growing degree-day since bubreak
        - **sun2scene**: PlantGl scene, when prodivided, a sun object (sphere) is added to it
        - **soil_size**: [cm] length of squared mesh size
    """
    print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    print '+ Project: ', wd
    print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    time_on = datetime.now()

    # Read user parameters
    if 'param_index' in kwargs:
	param_index_var = kwargs['param_index'] # loop through multiple parameter files
        params_path = wd + 'params%s.json' % param_index_var
	output_index = param_index_var
    else:
        params_path = wd + 'params.json'
	output_index = 1

    params = Params(params_path)
    #output_index = params.simulation.output_index
   

    # ==============================================================================
    # Initialisation
    # ==============================================================================
    #   Climate data
    meteo_path = wd + params.simulation.meteo
    meteo_tab = read_csv(meteo_path, sep=';', decimal='.', header=0)
    meteo_tab.time = DatetimeIndex(meteo_tab.time)
    meteo_tab = meteo_tab.set_index(meteo_tab.time)

    #   Adding missing data
    if 'Ca' not in meteo_tab.columns:
        meteo_tab['Ca'] = [400.] * len(meteo_tab)  # ppm [CO2]
    if 'Pa' not in meteo_tab.columns:
        meteo_tab['Pa'] = [101.3] * len(meteo_tab)  # atmospheric pressure

    #   Determination of the simulation period
    sdate = datetime.strptime(params.simulation.sdate, "%Y-%m-%d %H:%M:%S")
    edate = datetime.strptime(params.simulation.edate, "%Y-%m-%d %H:%M:%S")
    meteo = meteo_tab.ix[sdate:edate]
    eindex = meteo_tab.time.index.get_loc(edate) # make date_list 1 timepoint longer
    post_date = meteo_tab.time.ix[eindex+1] 
    date_list = meteo_tab.time.ix[sdate:post_date] # to calculate length of last timestep
    #datet = date_range(sdate, edate, freq='H')
    #meteo = meteo_tab.ix[datet]
    #time_conv = {'D': 86.4e3, 'H': 3600., 'T': 60., 'S': 1.}[datet.freqstr]
    time_conv = 3600.0    

    # Reading available pre-dawn soil water potential data
    if 'psi_soil' in kwargs:
        psi_pd = DataFrame([kwargs['psi_soil']] * len(meteo.time),
                           index=meteo.time, columns=['psi'])
	psi_soil = kwargs['psi_soil']
    elif 'initial_psi_soil' in kwargs: # only sets first timestep
	psi_soil = kwargs['initial_psi_soil'] # later steps calculated from water balance
    else:
        assert (isfile(wd + 'psi_soil.input')), "The 'psi_soil.input' file is missing."
        psi_pd = read_csv(wd + 'psi_soil.input', sep=';', decimal='.').set_index('time')
        psi_pd.index = [datetime.strptime(s, "%Y-%m-%d") for s in psi_pd.index]

    # Define irrigation dates
    irr_freq = 7 # weekly irrigation
    irr_freq_dt = timedelta(days=irr_freq) # irrigation period
    irr_sdate = sdate + irr_freq_dt # start irrigation after 1 period
    irr_remain = 0.0 # initialize irrigation 
    irr_to_apply = 0.0
    drip_rate = 3.8 # drip rate -2 emitters/vine at 0.5 gal/hr
    RDI = 0.6 # deficit irrigation replacement rate (0 to 1)
    dt_index = 0 # start at date 1  

    # Unit length conversion (from scene unit to the standard [m]) unit)
    unit_scene_length = params.simulation.unit_scene_length
    length_conv = {'mm': 1.e-3, 'cm': 1.e-2, 'm': 1.}[unit_scene_length]

    # Determination of cumulative degree-days parameter
    t_base = params.phenology.t_base
    budbreak_date = datetime.strptime(params.phenology.emdate, "%Y-%m-%d %H:%M:%S")

    if 'gdd_since_budbreak' in kwargs:
        gdd_since_budbreak = kwargs['gdd_since_budbreak']
    elif min(meteo_tab.index) <= budbreak_date:
        tdays = date_range(budbreak_date, sdate, freq='D')
        tmeteo = meteo_tab.ix[tdays].Tac.to_frame()
        tmeteo = tmeteo.set_index(DatetimeIndex(tmeteo.index).normalize())
        df_min = tmeteo.groupby(tmeteo.index).aggregate(np.min).Tac
        df_max = tmeteo.groupby(tmeteo.index).aggregate(np.max).Tac
        # df_tt = merge(df_max, df_min, how='inner', left_index=True, right_index=True)
        # df_tt.columns = ('max', 'min')
        # df_tt['gdd'] = df_tt.apply(lambda x: 0.5 * (x['max'] + x['min']) - t_base)
        # gdd_since_budbreak = df_tt['gdd'].cumsum()[-1]
        df_tt = 0.5 * (df_min + df_max) - t_base
        gdd_since_budbreak = df_tt.cumsum()[-1]
    else:
        raise ValueError('Cumulative degree-days temperature is not provided.')

    print 'GDD since budbreak = %d °Cd' % gdd_since_budbreak

    # Determination of perennial structure arms (for grapevine)
    # arm_vid = {g.node(vid).label: g.node(vid).components()[0]._vid for vid in g.VtxList(Scale=2) if
    #            g.node(vid).label.startswith('arm')}

    # Soil reservoir dimensions (inter row, intra row, depth) [m]
    soil_dimensions = params.soil.soil_dimensions
    soil_total_volume = soil_dimensions[0] * soil_dimensions[1] * soil_dimensions[2]
    rhyzo_coeff = params.soil.rhyzo_coeff
    rhyzo_total_volume = rhyzo_coeff * np.pi * min(soil_dimensions[:2]) ** 2 / 4. * soil_dimensions[2]

    # Counter clockwise angle between the default X-axis direction (South) and
    # the real direction of X-axis.
    scene_rotation = params.irradiance.scene_rotation

    # Sky and cloud temperature [degreeC]
    t_sky = params.energy.t_sky
    t_cloud = params.energy.t_cloud

    # Topological location
    latitude = params.simulation.latitude
    longitude = params.simulation.longitude
    elevation = params.simulation.elevation
    geo_location = (latitude, longitude, elevation)

    # Pattern
    ymax, xmax = map(lambda dim: dim / length_conv, soil_dimensions[:2])
    pattern = ((-xmax / 2.0, -ymax / 2.0), (xmax / 2.0, ymax / 2.0))

    # Label prefix of the collar internode
    vtx_label = params.mtg_api.collar_label

    # Label prefix of the leaves
    leaf_lbl_prefix = params.mtg_api.leaf_lbl_prefix

    # Label prefices of stem elements
    stem_lbl_prefix = params.mtg_api.stem_lbl_prefix

    E_type = params.irradiance.E_type
    tzone = params.simulation.tzone
    turtle_sectors = params.irradiance.turtle_sectors
    icosphere_level = params.irradiance.icosphere_level
    turtle_format = params.irradiance.turtle_format

    limit = params.energy.limit
    energy_budget = params.simulation.energy_budget
    solo = params.energy.solo
    simplified_form_factors = params.simulation.simplified_form_factors
    print 'Energy_budget: %s' % energy_budget

    # Optical properties
    opt_prop = params.irradiance.opt_prop

    print 'Hydraulic structure: %s' % params.simulation.hydraulic_structure

    psi_min = params.hydraulic.psi_min
    TLP = params.hydraulic.TLP

    # Parameters of leaf Nitrogen content-related models
    Na_dict = params.exchange.Na_dict

    # Computation of the form factor matrix
    form_factors=None
    if energy_budget:
        print 'Computing form factors...'
        if not simplified_form_factors:
            form_factors = energy.form_factors_matrix(g, pattern, length_conv, limit=limit)
        else:
            form_factors = energy.form_factors_simplified(g, pattern=pattern, infinite=True, leaf_lbl_prefix=leaf_lbl_prefix,
                                           turtle_sectors=turtle_sectors, icosphere_level=icosphere_level,
                                           unit_scene_length=unit_scene_length)



    # Soil class
    soil_class = params.soil.soil_class
    print 'Soil class: %s' % soil_class

    # Rhyzosphere concentric radii determination
    rhyzo_radii = params.soil.rhyzo_radii
    rhyzo_number = len(rhyzo_radii)

    # Add rhyzosphere elements to mtg
    rhyzo_solution = params.soil.rhyzo_solution
    print 'rhyzo_solution: %s' % rhyzo_solution


   # pdb.set_trace()
    if rhyzo_solution:
        dist_roots, rad_roots = params.soil.roots
        if not any(item.startswith('rhyzo') for item in g.property('label').values()):
            vid_collar = architecture.mtg_base(g, vtx_label=vtx_label)
            vid_base = architecture.add_soil_components(g, rhyzo_number, rhyzo_radii,
                                                        soil_dimensions, soil_class, vtx_label)
        else:
            vid_collar = g.node(g.root).vid_collar
            vid_base = g.node(g.root).vid_base

            radius_prev = 0.

            for ivid, vid in enumerate(g.Ancestors(vid_collar)[1:]):
                radius = rhyzo_radii[ivid]
                g.node(vid).Length = radius - radius_prev
                g.node(vid).depth = soil_dimensions[2] / length_conv  # [m]
                g.node(vid).TopDiameter = radius * 2.
                g.node(vid).BotDiameter = radius * 2.
                g.node(vid).soil_class = soil_class
                radius_prev = radius

    else:
        dist_roots, rad_roots = None, None
        # Identifying and attaching the base node of a single MTG
        vid_collar = architecture.mtg_base(g, vtx_label=vtx_label)
        vid_base = vid_collar

    g.node(g.root).vid_base = vid_base
    g.node(g.root).vid_collar = vid_collar

    # Initializing sapflow to 0
    for vtx_id in traversal.pre_order2(g, vid_base):
        g.node(vtx_id).Flux = 0.

    # Addition of a soil element
    if 'Soil' not in g.properties()['label'].values():
        if 'soil_size' in kwargs:
            if kwargs['soil_size'] > 0.:
                architecture.add_soil(g, kwargs['soil_size'])
        else:
            architecture.add_soil(g, 500.)

    # Suppression of undesired geometry for light and energy calculations
    geom_prop = g.properties()['geometry']
    vidkeys = []
    for vid in g.properties()['geometry']:
        n = g.node(vid)
        if not n.label.startswith(('L', 'other', 'soil')):
            vidkeys.append(vid)
    [geom_prop.pop(x) for x in vidkeys]
    g.properties()['geometry'] = geom_prop

    # Attaching optical properties to MTG elements
    g = irradiance.optical_prop(g, leaf_lbl_prefix=leaf_lbl_prefix,
                                stem_lbl_prefix=stem_lbl_prefix, wave_band='SW',
                                opt_prop=opt_prop)

    # Estimation of Nitroen surface-based content according to Prieto et al. (2012)
    # Estimation of intercepted irradiance over past 10 days:
    if not 'Na' in g.property_names():
        print 'Computing Nitrogen profile...'
        assert (sdate - min(
            meteo_tab.index)).days >= 10, 'Meteorological data do not cover 10 days prior to simulation date.'

        ppfd10_date = sdate + timedelta(days=-10)
        ppfd10t = date_range(ppfd10_date, sdate, freq='H')
        ppfd10_meteo = meteo_tab.ix[ppfd10t]
        caribu_source, RdRsH_ratio = irradiance.irradiance_distribution(ppfd10_meteo, geo_location, E_type,
                                                                        tzone, turtle_sectors, turtle_format,
                                                                        None, scene_rotation, None)

        # Compute irradiance interception and absorbtion
        g, caribu_scene = irradiance.hsCaribu(mtg=g,
                                              unit_scene_length=unit_scene_length,
                                              source=caribu_source, direct=False,
                                              infinite=True, nz=50, ds=0.5,
                                              pattern=pattern)

        g.properties()['Ei10'] = {vid: g.node(vid).Ei * time_conv / 10. / 1.e6 for vid in g.property('Ei').keys()}

        # Estimation of leaf surface-based nitrogen content:
        for vid in g.VtxList(Scale=3):
            if g.node(vid).label.startswith(leaf_lbl_prefix):
                g.node(vid).Na = exchange.leaf_Na(gdd_since_budbreak, g.node(vid).Ei10,
                                                  Na_dict['aN'],
                                                  Na_dict['bN'],
                                                  Na_dict['aM'],
                                                  Na_dict['bM'])
   
    # Define path to folder
    output_path = wd + 'output' + '/'
    
    # Save geometry in an external file
    #g.date = '10_days_sim'
    #architecture.mtg_save(g, scene, output_path) 
    #architecture.mtg_save_geometry(scene, output_path, '10_days_sim')
    #pdb.set_trace()
    # ==============================================================================
    # Simulations
    # ==============================================================================

    sapflow = []
    sapflow_tot = []
    irrigation_ls = []
    # sapEast = []
    # sapWest = []
    an_ls = []
    rg_ls = []
    psi_soil_ls = []
    psi_stem_ls = []
    psi_leaf_mean_ls = []
    psi_leaf_max_ls = []
    psi_leaf_min_ls = []
    pthresh_ls = []
    tthresh_ls = []
    psi_stem = {}
    Tlc_dict = {}
    Ei_dict = {}
    an_dict = {}
 #   LA_dict = {}
    gs_dict = {}

    # The time loop +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for date in meteo.time:
        print "=" * 72
        print 'Date', date, '\n'

        # Select of meteo data
        imeteo = meteo[meteo.time == date]

        # Add a date index to g
        g.date = datetime.strftime(date, "%Y%m%d%H%M%S")

	# Calculate length of current timestep
	ts_diff = date_list[dt_index+1] - date_list[dt_index]
	timestep_len = ts_diff.seconds # seconds between timesteps

        if 'sun2scene' not in kwargs or not kwargs['sun2scene']:
            sun2scene = None
        elif kwargs['sun2scene']:
            sun2scene = display.visu(g, def_elmnt_color_dict=True, scene=Scene())

        # Compute irradiance distribution over the scene
        caribu_source, RdRsH_ratio = irradiance.irradiance_distribution(imeteo, geo_location, E_type, tzone,
                                                                        turtle_sectors, turtle_format, sun2scene,
                                                                        scene_rotation, None)

        # Compute irradiance interception and absorbtion
        g, caribu_scene = irradiance.hsCaribu(mtg=g,
                                              unit_scene_length=unit_scene_length,
                                              source=caribu_source, direct=False,
                                              infinite=True, nz=50, ds=0.5,
                                              pattern=pattern)

        # g.properties()['Ei'] = {vid: 1.2 * g.node(vid).Ei for vid in g.property('Ei').keys()}

        # Trace intercepted irradiance on each time step
        rg_ls.append(sum([g.node(vid).Ei / (0.48 * 4.6) * surface(g.node(vid).geometry) * (length_conv ** 2) \
                          for vid in g.property('geometry') if g.node(vid).label.startswith('L')]))


        # Hack forcing of soil temperture (model of soil temperature under development)
        t_soil = energy.forced_soil_temperature(imeteo)

        # Climatic data for energy balance module
        # TODO: Change the t_sky_eff formula (cf. Gliah et al., 2011, Heat and Mass Transfer, DOI: 10.1007/s00231-011-0780-1)
        t_sky_eff = RdRsH_ratio * t_cloud + (1 - RdRsH_ratio) * t_sky

        solver.solve_interactions(g, imeteo, psi_soil, t_soil, t_sky_eff,
                                  vid_collar, vid_base, length_conv, timestep_len,
                                  rhyzo_total_volume, params, form_factors, simplified_form_factors)
        
        # Write mtg to an external file
        if scene is not None:
            architecture.mtg_save(g, scene, output_path)
            #architecture.mtg_save_geometry(scene, output_path, g.date)

        # Save results
        sapflow.append(g.node(vid_collar).Flux)
	sapflow_tot.append(g.node(vid_collar).Flux * timestep_len * 1000)
        # sapEast.append(g.node(arm_vid['arm1']).Flux)
        # sapWest.append(g.node(arm_vid['arm2']).Flux)
        an_ls.append(g.node(vid_collar).FluxC)
        psi_stem[date] = deepcopy(g.property('psi_head')) # water potentials
        Tlc_dict[date] = deepcopy(g.property('Tlc')) # temperature
        Ei_dict[date] = deepcopy(g.property('Eabs'))
        an_dict[date] = deepcopy(g.property('An')) # photosynthesis
	#LA_dict[date] = deepcopy(g.property('leaf_area')) # leaf area
	#Phot = np.asarray(an_dict[date].values())
	#Leaf_area = np.asarray(LA_dict[date].values())
	#An_LA = np.multiply(Phot, Leaf_area) # carbon gain per leaf
	#An_LA_tot = np.sum(An_LA) # total canopy carbon gain
	#An_LA_tot_ls.append(An_LA_tot) # store canopy carbon gain
        gs_dict[date] = deepcopy(g.property('gs')) # stomatal conductance	
	psi_stem_ls.append(g.node(3).psi_head) # collar WP 
	psi_leaf_mean_ls.append(np.mean([g.node(vid).psi_head for vid in g.property('gs').keys()])) # mean leaf WP
	psi_leaf_max_ls.append(np.amax([g.node(vid).psi_head for vid in g.property('gs').keys()])) # max leaf WP
	psi_leaf_min_ls.append(np.amin([g.node(vid).psi_head for vid in g.property('gs').keys()])) #min leaf WP

        # Calculate # leaves above a critical temperature threshold (47C)
        temp_array = np.asarray(Tlc_dict[date].values())
	temp_bool = temp_array > 47
	tthresh_ls.append(temp_bool.sum())

    	# Calculate # leaves below a critical WP threshold (TLP)
    	psi_array = [g.node(vid).psi_head for vid in g.property('gs').keys()]
    	psi_array2 = np.asarray(psi_array)
    	psi_bool = psi_array2 <= TLP
    	pthresh_ls.append(psi_bool.sum())

        # Read soil water potntial at midnight
        if 'psi_soil' in kwargs:
            psi_soil = kwargs['psi_soil']
	    irrigation_ls.append(0)
	elif 'initial_psi_soil' in kwargs:
	    # Estimate soil water potntial evolution due to transpiration
	   # pdb.set_trace()
            psi_soil_results_list = hydraulic.soil_water_potential_irrigated(psi_soil, irr_to_apply, irr_remain,  
                                                               date_list, dt_index, soil_class,  
                                                               soil_total_volume, irr_sdate, irr_freq,
                                                               RDI, drip_rate, sapflow_tot, psi_min)    
	    #pdb.set_trace()
	    psi_soil = psi_soil_results_list[0]
	    irr_remain = psi_soil_results_list[1]
	    irr_to_apply = psi_soil_results_list[2]
            irrigation_ls.append(psi_soil_results_list[3])
        else:
            if date.hour == 0:
                try:
                    psi_soil_init = psi_pd.ix[date.date()][0]
                    psi_soil = psi_soil_init
                except KeyError:
                    pass
            # Estimate soil water potntial evolution due to transpiration
            else:
                 psi_soil = hydraulic.soil_water_potential(psi_soil,
                                                          g.node(vid_collar).Flux * timestep_len,
                                                          soil_class, soil_total_volume, psi_min)
		
	psi_soil_ls.append(psi_soil)


        print '---------------------------'
        print 'psi_soil', round(psi_soil, 4)
        print 'psi_collar', round(g.node(3).psi_head, 4)
        print 'psi_leaf', round(np.median([g.node(vid).psi_head for vid in g.property('gs').keys()]), 4)
        print ''
        # print 'Rdiff/Rglob ', RdRsH_ratio
        # print 't_sky_eff ', t_sky_eff
        print 'gs', np.median(g.property('gs').values())
        print 'flux H2O', round(g.node(vid_collar).Flux * 1000. * timestep_len, 4)
        print 'flux C2O', round(g.node(vid_collar).FluxC, 4)
        print 'Tleaf ', round(np.median([g.node(vid).Tlc for vid in g.property('gs').keys()]), 2), \
            'Tair ', round(imeteo.Tac[0], 4)
        print ''
        print "=" * 72

        dt_index = dt_index +1

    # End time loop +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Write output
    # Plant total transpiration
    #sapflow = [flow * time_conv * 1000. for flow in sapflow]

    # sapEast, sapWest = [np.array(flow) * time_conv * 1000. for i, flow in enumerate((sapEast, sapWest))]

    # Mean, max, and min leaf temperature (C)
    t_mean = [np.mean(Tlc_dict[date].values()) for date in meteo.time]
    t_min = [np.amin(Tlc_dict[date].values()) for date in meteo.time]
    t_max = [np.amax(Tlc_dict[date].values()) for date in meteo.time]

    # Percentage of leaves above a critical temperature threshold (47C)
    tot_leaves = np.count_nonzero(temp_array) 
    tthresh_ls2 = np.array(tthresh_ls)  
    tthresh_ls_per = 100*tthresh_ls2/tot_leaves

    # Percentage of leaves below a critical water potential threshold (TLP)
    #psi_array = [g.node(vid).psi_head for vid in g.property('gs').keys()]
    #psi_array2 = np.asarray(psi_array)
    #tot_leaves_p = np.count_nonzero(psi_array2)
    #psi_bool = psi_array2 <= TLP
    pthresh_ls2 = np.array(pthresh_ls)
    pthresh_ls_per = 100*pthresh_ls2/tot_leaves


    # Mean, max, and min stomatal conductance (mol m-2 s-1)
    gs_mean = [np.mean(gs_dict[date].values()) for date in meteo.time]
    gs_max = [np.amax(gs_dict[date].values()) for date in meteo.time]
    gs_min = [np.amin(gs_dict[date].values()) for date in meteo.time]

    # Intercepted global radiation
    rg_ls = np.array(rg_ls) / (soil_dimensions[0] * soil_dimensions[1])

    results_dict = {
        'Rg': rg_ls,
        'An': an_ls,
        'E': sapflow,
        'Tleaf_mean': t_mean,
	'Tleaf_min': t_min,
	'Tleaf_max': t_max,
	'Tthresh': tthresh_ls_per,
        'Psi_soil': psi_soil_ls,
        'Psi_stem': psi_stem_ls,
        'Psi_leaf_mean': psi_leaf_mean_ls,
        'Psi_leaf_max': psi_leaf_max_ls,
        'Psi_leaf_min': psi_leaf_min_ls,
        'Pthresh': pthresh_ls_per,
        'Gs_mean': gs_mean,
        'Gs_max': gs_max,
        'Gs_min': gs_min,
	'Irrigation': irrigation_ls
    }

    # Results DataFrame
    results_df = DataFrame(results_dict, index=meteo.time)

    # Write
    if write_result:
        #results_df.to_csv(output_path + 'time_series.output',
        #              sep=';', decimal='.')
        results_df.to_csv('time_series_%s.output' % output_index,
                      sep=';', decimal='.')

    time_off = datetime.now()

    print ("")
    print ("beg time", time_on)
    print ("end time", time_off)
    print ("--- Total runtime: %d minute(s) ---" %
           int((time_off - time_on).seconds / 60.))

    return results_df
