from datetime import datetime, timedelta

import logging
#utils.setup_logging()
_log = logging.getLogger(__name__)

from vertex import Vertex
from helpers import *
from measurement_type import MeasurementType
from interval_value import IntervalValue
from meter_point import MeterPoint
from market_state import MarketState
from time_interval import TimeInterval
import matplotlib.pyplot as plt
import numpy as np

import itertools
import csv
import time
import numpy as np
import cvxpy
import xlrd
import os



class Market:
    # Market Base Class
    # A Market object may be a formal driver of myTransactiveNode's
    # responsibilities within a formal market. At least one Market must exist
    # (see the firstMarket object) to drive the timing with which new
    # TimeIntervals are created.
    
    def __init__(self, measurementType = [MeasurementType.PowerReal]):
        self.activeVertices = [[] for mt in measurementType]  # IntervalValue.empty  # values are vertices
        self.blendedPrices1 = []  # IntervalValue.empty  # future
        self.blendedPrices2 = []  # IntervalValue.empty  # future
        self.commitment = False
        self.converged = False
        self.defaultPrice = [0.05 for mt in measurementType]  # [$/kWh]
        self.electricity_rate = False
        self.gas_rate = False
        self.diesel_rate = False
        self.dualCosts = [[] for mt in measurementType]   # IntervalValue.empty  # values are [$]
        self.dualityGapThreshold = 0.01  # [dimensionless, 0.01 = 1#]
        self.futureHorizon = timedelta(hours=24)  # [h]                                          # [h]
        self.initialMarketState = MarketState.Inactive  # enumeration
        self.intervalDuration = timedelta(hours=1)  # [h]
        self.intervalsToClear = 1  # postitive integer
        self.marginalPrices = [[] for mt in measurementType]   # IntervalValue.empty  # values are [$/kWh]
        self.marketClearingInterval = timedelta(hours=1)  # [h]
        self.marketClearingTime = []  # datetime.empty  # when market clears
        self.measurementType = measurementType # types of energy that must be balanced on this market
        self.method = 2  # Calculation method {1: subgradient, 2: interpolation}
        self.marketOrder = 1  # ordering of sequential markets [pos. integer]
        self.name = ''
        self.netPowers = [[] for mt in measurementType]   # IntervalValue.empty  # values are [avg.kW]
        self.nextMarketClearingTime = []  # datetime.empty  # start of pattern
        self.productionCosts = [[] for mt in measurementType]  # IntervalValue.empty  # values are [$]
        self.timeIntervals = []  # TimeInterval.empty  # struct TimeInterval
        self.totalDemand = [[] for mt in measurementType]  # IntervalValue.empty  # [avg.kW]
        self.totalDualCost = [0.0 for mt in measurementType]*len(measurementType)  # [$]
        self.totalGeneration = [[] for mt in measurementType]   # IntervalValue.empty  # [avg.kW]
        self.totalProductionCost = [[] for mt in measurementType]   # [$]

    def assign_system_vertices(self, mtn):
        # FUNCTION ASSIGN_SYSTEM_VERTICES() - Collect active vertices from neighbor
        # and asset models and reassign them with aggregate system information for
        # all active time intervals.
        #
        # ASSUMPTIONS:
        # - Active time intervals exist and are up-to-date
        # - Local convergence has occurred, meaning that power balance, marginal
        # price, and production costs have been adequately resolved from the
        # local agent's perspective
        # - The active vertices of local asset models exist and are up-to-date.
        # The vertices represent available power flexibility. The vertices
        # include meaningful, accurate production-cost information.
        # - There is agreement locally and in the network concerning the format
        # and content of transactive records
        # - Calls method mkt.sum_vertices in each time interval.
        #
        # INPUTS:
        # mkt - Market object
        # mtn - myTransactiveNode object
        #
        # OUTPUTS:
        # - Updates mkt.activeVertices - vertices that define the net system
        # balance and flexibility. The meaning of the vertex properties are
        # - marginalPrice: marginal price [$/kWh]
        # - cost: total production cost at the vertex [$]. (A locally
        #   meaningful blended electricity price is (total production cost /
        #   total production)).
        # - power: system net power at the vertex (The system "clears" where
        #   system net power is zero.)
        if hasattr(self, 'measurementType'):
            n_power_types = len(self.measurementType)
        else:
            n_power_types = 1

        for ti in self.timeIntervals:
            # iterate through energy types
            for i_energy_type in range(n_power_types):
                if hasattr(self, 'measurementType'):
                    this_energy_type = self.measurementType[i_energy_type]
                else:
                    this_energy_type = MeasurementType.PowerReal                

                # Find and delete existing aggregate active vertices in the indexed
                # time interval. These shall be recreated.
                #ind = ~ismember([mkt.activeVertices.timeInterval], ti[i])  # logical array
                #mkt.activeVertices = mkt.activeVertices(ind)  # IntervalValues
                self.activeVertices[i_energy_type] = [x for x in self.activeVertices[i_energy_type] if x != ti]

                # Call the utility method mkt.sum_vertices to recreate the
                # aggregate vertices in the indexed time interval. (This method is
                # separated out because it will be used by other methods.)
                s_vertices = self.sum_vertices(mtn, ti, energy_type = this_energy_type)

                # Create and store interval values for each new aggregate vertex v
                # if you have multiple energy types, then save them separately
                for sv in s_vertices:
                    iv = IntervalValue(self, ti, self, MeasurementType.SystemVertex, sv) # an IntervalValue
                    self.activeVertices[i_energy_type].append(iv)


    def centralized_dispatch(self, mtn):
        ############### Linear Programming WSU Dispatcher #################
        ###################################################################
        # The intention of this script is to create dispatch setpoints for
        # the gas turbines, boilers, and chillers on WSU's campus to meet
        # electrical and thermal loads. These dispatches are not optimal
        # given the poor fidelity with which they model component efficiency
        # curves. These dispatches also do not include the dispatch of the
        # cold water energy storage tank on campus. All buildings are assumed
        # to have fixed loads which are the inputs. This script does not
        # include unit commitment.

        # This script follows the sequence below:
        # 0) read in load signal from csv?
        # 1) create linear efficiency fits from component data
        # 2) variables are defined and added to problem
        # 3) constraint functions are defined and added to problem
        # 4) objective function is defined and added to problem

        # This script was written by Nadia Panossian at Washington State University
        # and was last updated by:
        # Nadia Panossian on 10/14/2019
        # the author can be reached at nadia.panossian@wsu.edu

        RANGE = -1
        allow_thermal_slack = False
        T = self.intervalsToClear
        dt = self.intervalDuration.total_seconds()/3600

        global var_name_list
        var_name_list = []

        class VariableGroup(object):
            def __init__(self, name, indexes=(), is_binary_var=False, lower_bound_func=None, upper_bound_func=None, T=T, pieces=[1]):
                self.variables = {}

                name_base = name
                # if it is a piecewise function, make the variable group be a group of arrays (1,KK)
                if pieces == [1]:
                    pieces = [1 for i in indexes[0]]

                # create name base string
                for _ in range(len(indexes)):
                    name_base += "_{}"

                # create variable for each timestep and each component with a corresponding name
                for index in itertools.product(*indexes):
                    var_name = name_base.format(*index)

                    if is_binary_var:
                        var = binary_var(var_name)
                    else:
                        # assign upper and lower bounds for the variable
                        if lower_bound_func is not None:
                            lower_bound = lower_bound_func(index)
                        else:
                            lower_bound = None

                        if upper_bound_func is not None:
                            upper_bound = upper_bound_func(index)
                        else:
                            upper_bound = None

                        # the lower bound should always be set if the upper bound is set
                        if lower_bound is None and upper_bound is not None:
                            raise RuntimeError("Lower bound should not be unset while upper bound is set")

                        # create the cp variable
                        if lower_bound_func == constant_zero:
                            var = cvxpy.Variable(pieces[index[0]], name=var_name, nonneg=True)
                        elif lower_bound is not None:
                            var = cvxpy.Variable(pieces[index[0]], name=var_name)
                            # constr = [var>=lower_bound]
                        elif upper_bound is not None:
                            var = cvxpy.Variable(pieces[index[0]], name=var_name)
                            # constr = [var<=upper_bound, var>=lower_bound]
                        else:
                            var = cvxpy.Variable(pieces[index[0]], name=var_name)

                    self.variables[index] = var
                    var_name_list.append(var_name)
                    # self.constraints[index] = constr

            # internal function to find variables associated with your key
            def match(self, key):
                position = key.index(RANGE)

                def predicate(xs, ys):
                    z = 0
                    for i, (x, y) in enumerate(zip(xs, ys)):
                        if i != position and x == y:
                            z += 1
                    return z == len(key) - 1

                keys = list(self.variables.keys())
                keys = [k for k in keys if predicate(k, key)]
                keys.sort(key=lambda k: k[position])

                return [self.variables[k] for k in keys]

            # variable function to get the variables associated with the key
            def __getitem__(self, key):
                if type(key) != tuple:
                    key = (key,)

                n_range = key.count(RANGE)

                if n_range == 0:
                    return self.variables[key]
                elif n_range == 1:
                    return self.match(key)
                else:
                    raise ValueError("Can only get RANGE for one index.")

        def constant(x):
            def _constant(*args, **kwargs):
                return x

            return _constant


        ############# read in demand #########################################
        wb = xlrd.open_workbook(os.getcwd() + '/wsu_campus_2009_2012.xlsx')
        dem_sheet = wb.sheet_by_index(0)
        weather_sheet = wb.sheet_by_index(1)

        tdb = []  # dry bulb temp
        irrad_dire_norm = []  # direct normal irradiation
        e = []
        h = []
        c = []
        for i in range(1, T + 1):
            e.append(dem_sheet.cell_value(i, 0))
            h.append(dem_sheet.cell_value(i, 1))
            c.append(dem_sheet.cell_value(i, 2))

            tdb.append(weather_sheet.cell_value(i, 0))
            irrad_dire_norm.append(weather_sheet.cell_value(i, 1))

        # heat has way more nans, so remove them
        h = h[:37810]

        # demand has one value for entire campus
        demand = {'e': e, 'h': h, 'c': c}
        weather = {'t_db': tdb, 'irrad_dire_norm': irrad_dire_norm}


        n_boilers = 0
        n_chillers = 0
        n_turbines = 0
        n_dieselgen = 0
        n_flexible_building = 0
        n_inflexible_building = 0

        boiler_size = []
        chiller_size = []
        turbine_size = []

        boiler_eff = []
        chiller_eff =[]
        turbine_eff = []
        for asset in mtn.localAssets:
            if 'boiler' in str(asset.model):
                boiler_size.append(asset.model.activeVertices[str(MeasurementType.Heat)][0].value[2].power)
                boiler_eff.append(asset.model.activeVertices[str(MeasurementType.Heat)][0].value[2].power * self.gas_rate / asset.model.activeVertices[str(MeasurementType.Heat)][0].value[2].cost)
                n_boilers = n_boilers+1
            elif 'chiller' in str(asset.model):
                chiller_size.append(asset.model.activeVertices[str(MeasurementType.Cooling)][0].value[2].power)
                chiller_eff.append(asset.model.activeVertices[str(MeasurementType.Cooling)][0].value[2].power * self.electricity_rate / asset.model.activeVertices[str(MeasurementType.Cooling)][0].value[2].cost)
                n_chillers = n_chillers+1
            elif 'turbine' in str(asset.model):
                turbine_size.append(asset.model.activeVertices[str(MeasurementType.PowerReal)][0].value[2].power)
                turbine_eff.append(asset.model.activeVertices[str(MeasurementType.PowerReal)][0].value[2].power * self.gas_rate / asset.model.activeVertices[str(MeasurementType.PowerReal)][0].value[2].cost)
                n_turbines = n_turbines+1
            elif 'diesel' in str(asset.model):
                n_dieselgen = n_dieselgen+1
            elif 'InflexibleBuilding' in str(asset.model):
                n_inflexible_building = n_inflexible_building+1
            elif 'FlexibleBuilding' in str(asset.model):
                n_flexible_building = n_flexible_building+1

        inflex_elec_neutral = [[] for i in range(n_inflexible_building)]
        inflex_cool_neutral = [[] for i in range(n_inflexible_building)]
        inflex_heat_neutral = [[] for i in range(n_inflexible_building)]
        ninfb = 0
        for asset in mtn.localAssets:
            if 'InflexibleBuilding' in str(asset.model):
                for t in range(T):
                    inflex_elec_neutral[ninfb].append(asset.model.activeVertices[str(MeasurementType.PowerReal)][t].value.power)
                    inflex_cool_neutral[ninfb].append(asset.model.activeVertices[str(MeasurementType.Cooling)][t].value.power)
                    inflex_heat_neutral[ninfb].append(asset.model.activeVertices[str(MeasurementType.Heat)][t].value.power)
                ninfb = ninfb + 1


        temp_neutral = [[] for i in range(n_flexible_building)]
        temp_up = [[] for i in range(n_flexible_building)]
        temp_lower = [[] for i in range(n_flexible_building)]
        elec_neutral = [[] for i in range(n_flexible_building)]
        cool_neutral = [[] for i in range(n_flexible_building)]
        heat_neutral = [[] for i in range(n_flexible_building)]
        cool_v_temp = [[] for i in range(n_flexible_building)]
        heat_v_temp = [[] for i in range(n_flexible_building)]
        elec_v_temp = [[] for i in range(n_flexible_building)]
        cost_v_temp = np.zeros((n_flexible_building, T))

        nfb = 0
        for asset in mtn.localAssets:
            if 'FlexibleBuilding' in str(asset.model):
                for t in range(T):
                    # electrical loads
                    elec_neutral[nfb].append(asset.model.activeVertices[str(MeasurementType.PowerReal)][t].value[0].power)
                    elec_up = asset.model.activeVertices[str(MeasurementType.PowerReal)][t].value[0].power - asset.model.activeVertices[str(MeasurementType.PowerReal)][t].value[0].power
                    elec_lower = asset.model.activeVertices[str(MeasurementType.PowerReal)][t].value[0].power - asset.model.activeVertices[str(MeasurementType.PowerReal)][t].value[0].power

                    # make lists for boundaries on temperature
                    temp_neutral[nfb].append(asset.model.activeVertices[str(MeasurementType.Heat)][t].value[0].marginalPrice)
                    temp_lower[nfb].append(asset.model.activeVertices[str(MeasurementType.Heat)][t].value[1].marginalPrice - asset.model.activeVertices[str(MeasurementType.Heat)][t].value[0].marginalPrice)
                    temp_up[nfb].append(asset.model.activeVertices[str(MeasurementType.Heat)][t].value[2].marginalPrice - asset.model.activeVertices[str(MeasurementType.Heat)][t].value[0].marginalPrice)

                    # make list loads at each temperature setpoint

                    # cooling loads
                    cool_neutral[nfb].append(asset.model.activeVertices[str(MeasurementType.Cooling)][t].value[0].power)
                    cool_lower = asset.model.activeVertices[str(MeasurementType.Cooling)][t].value[1].power - asset.model.activeVertices[str(MeasurementType.Cooling)][t].value[0].power
                    cool_up = asset.model.activeVertices[str(MeasurementType.Cooling)][t].value[2].power - asset.model.activeVertices[str(MeasurementType.Cooling)][t].value[0].power


                    # heating loads
                    heat_neutral[nfb].append(asset.model.activeVertices[str(MeasurementType.Heat)][t].value[0].power)
                    heat_lower = asset.model.activeVertices[str(MeasurementType.Heat)][t].value[1].power - asset.model.activeVertices[str(MeasurementType.Heat)][t].value[0].power
                    heat_up = asset.model.activeVertices[str(MeasurementType.Heat)][t].value[2].power - asset.model.activeVertices[str(MeasurementType.Heat)][t].value[0].power


                    # discomfort cost
                    cost_up = asset.model.activeVertices[str(MeasurementType.Heat)][t].value[1].cost
                    cost_down = asset.model.activeVertices[str(MeasurementType.Heat)][t].value[2].cost

                    # make linear relationship between heating, cooling, electrical, and temperature setpoint
                    cool_v_temp[nfb].append((cool_lower / temp_lower[nfb][t] + cool_up / temp_up[nfb][t]) / 2)
                    heat_v_temp[nfb].append((heat_lower / temp_lower[nfb][t] + heat_up / temp_up[nfb][t]) / 2)
                    elec_v_temp[nfb].append((elec_lower / temp_lower[nfb][t] + elec_up / temp_up[nfb][t]) / 2)
                    # make linear relationship between temperature difference and cost
                    cost_v_temp[nfb, t] = (cost_up / temp_up[nfb][t] - cost_down / temp_lower[nfb][t]) / 2

                nfb = nfb+1

        constant_zero = constant(0)
        index_hour = (range(T),)
        index_nodes = (range(1), range(T))

        ep_elecfromgrid = VariableGroup("ep_elecfromgrid", indexes=index_nodes,lower_bound_func=constant_zero)  # real power from grid
        ep_electogrid = VariableGroup("ep_electogrid", indexes=index_nodes,lower_bound_func=constant_zero)  # real power to the grid
        elec_unserve = VariableGroup("elec_unserve", indexes=index_nodes, lower_bound_func=constant_zero)

        if n_boilers > 0:
            heat_unserve = VariableGroup("heat_unserve", indexes=index_nodes, lower_bound_func=constant_zero)

            heat_dump = VariableGroup("heat_dump", indexes=index_nodes, lower_bound_func=constant_zero)
        if n_chillers > 0:
            cool_unserve = VariableGroup("cool_unserve", indexes=index_nodes, lower_bound_func=constant_zero)

        # turbines: # fuel cells are considered turbines
        index_turbines = range(n_turbines), range(24)
        turbine_y = VariableGroup("turbine_y", indexes=index_turbines, lower_bound_func=constant_zero)  # fuel use
        turbine_xp = VariableGroup("turbine_xp", indexes=index_turbines, lower_bound_func=constant_zero)  # real power output

        # diesel generators
        index_dieselgen = range(n_dieselgen), range(24)
        dieselgen_y = VariableGroup("dieselgen_y", indexes=index_dieselgen, lower_bound_func=constant_zero)  # fuel use
        dieselgen_xp = VariableGroup("dieselgen_xp", indexes=index_dieselgen, lower_bound_func=constant_zero) # real power output

        # boilers:
        index_boilers = range(n_boilers), range(24)
        boiler_y = VariableGroup("boiler_y", indexes=index_boilers, lower_bound_func=constant_zero)  # fuel use from boiler
        boiler_x = VariableGroup("boiler_x", indexes=index_boilers, lower_bound_func=constant_zero)  # heat output from boiler

        # chillers
        index_chiller = range(n_chillers), range(24)
        chiller_x = VariableGroup("chiller_x", indexes=index_chiller, lower_bound_func=constant_zero)  # cooling power output
        chiller_yp = VariableGroup("chiller_yp", indexes=index_chiller, lower_bound_func=constant_zero)  # real electric power demand


        # buildingn temperatuers
        index_temp = range(n_flexible_building), range(T)
        temp = VariableGroup("temp", indexes=index_temp)

        constraints = []

        def add_constraint(name, indexes, constraint_func):
            name_base = name
            for _ in range(len(indexes)):
                name_base += "_{}"

            for index in itertools.product(*indexes):
                name = name_base.format(*index)
                c = constraint_func(index)
                constraints.append((c, name))

        def electric_p_balance(index):
            i, t = index
            # sum of power
            return cvxpy.sum([turbine_xp[j, t] for j in range(n_turbines)]) \
                   + ep_elecfromgrid[0, t] - ep_electogrid[0, t] \
                   - cvxpy.sum([chiller_yp[j, t] for j in range(n_chillers)]) \
                   + cvxpy.sum([dieselgen_xp[j, t] for j in range(n_dieselgen)]) \
                   - cvxpy.sum([inflex_elec_neutral[j][t] for j in range(n_inflexible_building)]) \
                   - cvxpy.sum([elec_neutral[j][t] for j in range(n_flexible_building)]) \
                   - cvxpy.sum([elec_v_temp[j][t] * temp[j, t] for j in range(n_flexible_building)]) \
                   + elec_unserve[0, t] == 0

        def heat_balance(index):
            i, t = index
            # sum of heat produced-heat used at this node = heat in/out of this node
            return cvxpy.sum([boiler_x[j, t] for j in range(n_boilers)]) \
                   + cvxpy.sum([(1 - turbine_eff[j]) * turbine_xp[j, t] for j in range(n_turbines)]) \
                   - cvxpy.sum([inflex_heat_neutral[j][t] for j in range(n_inflexible_building)]) \
                   - cvxpy.sum([heat_neutral[j][t] for j in range(n_flexible_building)]) \
                   - cvxpy.sum([heat_v_temp[j][t] * temp[j, t] for j in range(n_flexible_building)]) \
                   - heat_dump[0, t] \
                   + heat_unserve[0, t] \
                   == 0

        def cool_balance(index):
            i, t = index
            return cvxpy.sum([chiller_x[j, t] for j in range(n_chillers)]) \
                   + cool_unserve[0, t] \
                   - cvxpy.sum([inflex_cool_neutral[j][t] for j in range(n_inflexible_building)]) \
                   - cvxpy.sum([cool_neutral[j][t] for j in range(n_flexible_building)]) \
                   - cvxpy.sum([cool_v_temp[j][t] * temp[j, t] for j in range(n_flexible_building)]) \
                   == 0

        def turbine_y_consume(index):
            i, t = index
            return turbine_xp[i, t] / turbine_eff[i] - turbine_y[i, t] == 0

        def turbine_xp_upper(index):
            i, t = index
            return turbine_xp[i, t] <= turbine_size[i]

        def dieselgen_y_consume(index):
            i, t = index
            return dieselgen_xp[i, t] / diesel_eff[i] - dieselgen_y[i, t] == 0

        def dieselgen_xp_upper(index):
            i, t = index
            return dieselgen_xp[i, t] <= diesel_size[i]

        def boiler_y_consume(index):
            i, t = index
            return boiler_x[i, t] / boiler_eff[i] - boiler_y[i, t] == 0

        def boiler_x_upper(index):
            i, t = index
            return boiler_x[i, t] <= boiler_size[i]

        def chiller_yp_consume(index):
            i, t = index
            return chiller_x[i, t] / chiller_eff[i] - chiller_yp[i, t] == 0

        def chiller_x_upper(index):
            i, t = index
            return chiller_x[i, t] <= chiller_size[i]

        def no_slack_c(index):
            j, t = index
            return cool_unserve[j, t] == 0

        def no_slack_h(index):
            j, t = index
            return heat_unserve[j, t] == 0

        def no_slack_e(index):
            j, t = index
            return elec_unserve[j, t] == 0

        def temperature_ub(index):
            j, t = index
            return temp[j, t] <= temp_up[j][t]

        def temperature_lb(index):
            j, t = index
            return temp[j, t] >= temp_lower[j][t]

        add_constraint("electric_p_balance", index_nodes, electric_p_balance)
        add_constraint("heat_balance", index_nodes, heat_balance)
        add_constraint("cool_balance", index_nodes, cool_balance)

        # add turbine constraints
        index_turbine = (range(n_turbines),)
        add_constraint("turbine_y_consume", index_turbine + index_hour, turbine_y_consume)  # False
        add_constraint("turbine_xp_upper", index_turbine + index_hour, turbine_xp_upper)

        # add diesel constraints
        index_diesel = (range(n_dieselgen),)
        add_constraint("dieselgen_y_consume", index_diesel + index_hour, dieselgen_y_consume)
        add_constraint("dieselgen_xp_upper", index_diesel + index_hour, dieselgen_xp_upper)

        # add boiler constraints
        index_boiler = (range(n_boilers),)
        add_constraint("boiler_y_consume", index_boiler + index_hour, boiler_y_consume)
        add_constraint("boiler_x_upper", index_boiler + index_hour, boiler_x_upper)

        # add chiller constriants
        index_chiller = (range(n_chillers),)
        add_constraint("chiller_yp_consume", index_chiller + index_hour, chiller_yp_consume)
        add_constraint("chiller_x_upper", index_chiller + index_hour, chiller_x_upper)

        if allow_thermal_slack == False:
            add_constraint("no_slack_h", index_nodes, no_slack_h)
            add_constraint("no_slack_c", index_nodes, no_slack_c)
            add_constraint("no_slack_e", index_nodes, no_slack_e)

        # add building temperature constraints
        add_constraint("temperature_ub", index_temp, temperature_ub)
        add_constraint("temperature_lb", index_temp, temperature_lb)

        ##################### add objective functions ################################

        objective_components = []

        # utility elec cost
        for var in ep_elecfromgrid[0, RANGE]:
            objective_components.append(var * self.electricity_rate)

        # gas for gas turbines
        for i in range(n_turbines):
            for var in turbine_y[i, RANGE]:
                objective_components.append(var * self.gas_rate)

        # diesel for diesel generators
        for i in range(n_dieselgen):
            for var in dieselgen_y[i, RANGE]:
                objective_components.append(var * self.diesel_rate)

        # gas for boilers
        for i in range(n_boilers):
            for var in boiler_y[i, RANGE]:
                objective_components.append(var * self.gas_rate)

        # discomfort cost
        for i in range(n_flexible_building):
            for var, discomfort_cost in zip(temp[i, RANGE], cost_v_temp[i, :]):
                #discomfort_cost = 0
                objective_components.append(cvxpy.abs(var) * discomfort_cost)

        ####################### create and solve the problem ############################

        objective = cvxpy.Minimize(cvxpy.sum(objective_components))
        constraints_list = [x[0] for x in constraints]
        prob = cvxpy.Problem(objective, constraints_list)
        print('problem created, solving problem')

        tic = time.time()

        result = prob.solve(solver='ECOS')

        toc = time.time() - tic
        print('optimal cost: ' + str(result))
        print('problem solved in ' + str(toc) + 'seconds')

        values = {}
        turbine_dispatch = []
        dieselgen_dispatch = []
        boiler_dispatch = []
        chiller_dispatch = []
        flexible_building_dispatch = []

        for i in range(len(var_name_list)):
            var_name = var_name_list[i]
            split_name = var_name.split('_')
            var_name = var_name.split(split_name[-2])[0][:-1]
            idx = int(split_name[-2])
            step = int(split_name[-1])
            if step == 0:
                field_name = var_name + '_' + str(idx)
                var_val = eval(var_name)[idx, step]
                if var_val.attributes['boolean']:
                    var_val = var_val.value
                elif var_val.value == None:
                    var_val = 0
                else:
                    var_val = var_val.value[0]

                if (var_name == 'turbine_xp'):
                    turbine_dispatch.append(var_val)
                elif (var_name == 'dieselgen_xp'):
                    dieselgen_dispatch.append(var_val)
                elif (var_name == 'boiler_x'):
                    boiler_dispatch.append(var_val)
                elif (var_name == 'chiller_x'):
                    chiller_dispatch.append(var_val)
                elif (var_name == 'temp'):
                    flexible_building_dispatch.append(var_val)

        boiler_idx = 0
        chiller_idx = 0
        turbine_idx = 0
        dieselgen_idx = 0
        flexible_building_idx = 0
        inflexible_building_idx = 0
        for asset in mtn.localAssets:
            if 'boiler' in str(asset.model):
                asset.model.scheduledPowers[0] = boiler_dispatch[boiler_idx]
                boiler_idx = boiler_idx+1
            elif 'chiller' in str(asset.model):
                asset.model.scheduledPowers[0] = chiller_dispatch[chiller_idx]
                chiller_idx = chiller_idx + 1
            elif 'turbine' in str(asset.model):
                asset.model.scheduledPowers[0] = turbine_dispatch[turbine_idx]
                turbine_idx = turbine_idx + 1
            elif 'diesel' in str(asset.model):
                asset.model.scheduledPowers[0] = dieselgen_dispatch[dieselgen_idx]
                diesel_idx = dieselgen_idx+1
            elif 'FlexibleBuilding' in str(asset.model):
                T_setpoint = flexible_building_dispatch[flexible_building_idx]
                asset.model.scheduledPowers[0] = T_setpoint + temp_neutral[flexible_building_idx][0]
                if T_setpoint <= 0:
                    asset.model.scheduledPowers[1] = heat_neutral[flexible_building_idx][0] + \
                    (asset.model.activeVertices[str(MeasurementType.Heat)][0].value[1].power - heat_neutral[flexible_building_idx][0])*T_setpoint/temp_lower[flexible_building_idx][0]
                    asset.model.scheduledPowers[2] = cool_neutral[flexible_building_idx][0] + \
                    (asset.model.activeVertices[str(MeasurementType.Cooling)][0].value[1].power - cool_neutral[flexible_building_idx][0])*T_setpoint/temp_lower[flexible_building_idx][0]
                else:
                    asset.model.scheduledPowers[1] = heat_neutral[flexible_building_idx][0] + \
                    (asset.model.activeVertices[str(MeasurementType.Heat)][0].value[2].power - heat_neutral[flexible_building_idx][0])*T_setpoint/temp_lower[flexible_building_idx][0]
                    asset.model.scheduledPowers[2] = cool_neutral[flexible_building_idx][0] + \
                    (asset.model.activeVertices[str(MeasurementType.Cooling)][0].value[2].power - cool_neutral[flexible_building_idx][0])*T_setpoint/temp_lower[flexible_building_idx][0]
                flexible_building_idx = flexible_building_idx+1
            elif 'InflexibleBuilding' in str(asset.model):
                asset.model.scheduledPowers[0] = inflex_elec_neutral[inflexible_building_idx][0]
                asset.model.scheduledPowers[1] = inflex_heat_neutral[inflexible_building_idx][0]
                asset.model.scheduledPowers[2] = inflex_cool_neutral[inflexible_building_idx][0]
                inflexible_building_idx = inflexible_building_idx+1

        asset_type = ['FlexibleBuilding']
        #self.view_vertices_and_schedules(mtn, asset_type)
        print('One Time step solved and vertices are updated with schedules')

    def view_vertices_and_schedules(self, mtn, asset_type):

        for n_type in range(len(asset_type)):
            number_assets = 0
            for asset in mtn.localAssets:
                if (asset_type[n_type] in str(asset.model)) and (asset_type[n_type] == 'FlexibleBuilding'):
                    number_assets = number_assets+2 # one plot for heating and one plot for cooling
                elif asset_type[n_type] in str(asset.model):
                    number_assets = number_assets + 1
            plt.figure()
            if number_assets < 3:
                subplot_rows = number_assets
                fig, axs = plt.subplots(subplot_rows)
            else:
                subplot_rows == number_assets/2 + number_assets%2
                fig, axs = plt.subplots(subplot_rows,2)

            for asset in mtn.localAssets:
                if asset_type[n_type] in str(asset.model) and (asset_type[n_type] == 'FlexibleBuilding'):

                    cool_neutral = asset.model.activeVertices[str(MeasurementType.Cooling)][0].value[0].power
                    cool_lower = asset.model.activeVertices[str(MeasurementType.Cooling)][0].value[1].power
                    cool_up = asset.model.activeVertices[str(MeasurementType.Cooling)][0].value[2].power

                    cool_y = [cool_lower, cool_neutral, cool_up]

                    heat_neutral = asset.model.activeVertices[str(MeasurementType.Heat)][0].value[0].power
                    heat_lower = asset.model.activeVertices[str(MeasurementType.Heat)][0].value[1].power
                    heat_up = asset.model.activeVertices[str(MeasurementType.Heat)][0].value[2].power

                    heat_y = [heat_lower, heat_neutral, heat_up]

                    temp_neutral = asset.model.activeVertices[str(MeasurementType.Heat)][0].value[0].marginalPrice
                    temp_lower = asset.model.activeVertices[str(MeasurementType.Heat)][0].value[1].marginalPrice
                    temp_up = asset.model.activeVertices[str(MeasurementType.Heat)][0].value[2].marginalPrice

                    temp_x = [temp_lower, temp_neutral, temp_up]

                    for i in range(len(asset.model.measurementType)):
                        if asset.model.measurementType[i] == MeasurementType.PowerReal:
                            Tset_dispatched = asset.model.scheduledPowers[i]
                        elif asset.model.measurementType[i] == MeasurementType.Heat:
                            heat_dispatched = asset.model.scheduledPowers[i]
                        elif asset.model.measurementType[i] == MeasurementType.Cooling:
                            cool_dispatched = asset.model.scheduledPowers[i]


                    axs[0].plot(temp_x, cool_y)
                    axs[0].plot([Tset_dispatched,Tset_dispatched], [0,cool_dispatched], color='red', marker='o')
                    axs[0].set_title('Axis [0,0]')
                    axs[1].plot(temp_x, heat_y)
                    axs[1].plot([Tset_dispatched, Tset_dispatched], [0, heat_dispatched], color='red', marker='o')
                    axs[1].set_title('Axis [0,0]')


            plt.show()










    def check_intervals(self):
        # FUNCTION CHECK_INTERVALS()
        # Check or create the set of instantiated TimeIntervals in this Market
        #
        # mkt - Market object
        
        # Create the array "steps" of time intervals that should be active.
        # NOTE: Function Hours() corrects the behavior of Matlab function
        # hours().
        #steps = datetime(mkt.marketClearingTime): Hours(mkt.marketClearingInterval): datetime + Hours(mkt.futureHorizon)
        #steps = steps(steps > datetime - Hours(mkt.marketClearingInterval))
        steps = []
        cur_time = datetime.now()
        end_time = cur_time + self.futureHorizon
        step_time = self.marketClearingTime
        while step_time < end_time:
            if step_time > cur_time - self.marketClearingInterval:
                steps.append(step_time)
            step_time = step_time + self.marketClearingInterval
            
        # Index through the needed TimeIntervals based on their start times.
        for i in range(len(steps)):  #for i = 1:len(steps)
            # This is a test to see whether the interval exists.
            # Case 0: a new interval must be created
            # Case 1: There is one match, the TimeInterval exists
            # Otherwise: Duplicates exists and should be deleted.

            #switch len(findobj(mkt.timeIntervals, 'startTime', steps(i)))
            tis = [x for x in self.timeIntervals if x.startTime == steps[i]]
            tis_len = len(tis)
            
            # No match was found. Create a new TimeInterval.
            if tis_len == 0:
                
                # Create the TimeInterval
                # Modified 1/29 to use TimeInterval constructor
                at = steps[i] - self.futureHorizon  # activationTime
                dur = self.intervalDuration  # duration
                mct = steps[i]  # marketClearingTime
                st = steps[i]  # startTime
                
                ti = TimeInterval(at, dur, self, mct, st)
                
                # ELIMINATE HURKY INLINE CONSTRUCTION BELOW
                # ti = TimeInterval()
                #
                #     #Populate the TimeInterval properties
                #
                #     ti.name = char(steps[i],'yyMMdd-hhmm')
                #     ti.startTime = steps[i]
                #     ti.active = True
                #     ti.duration = Hours(mkt.intervalDuration)
                #     ti.marketClearingTime = steps[i]
                #     ti.market = mkt
                #     ti.activationTime = steps[i]-Hours(mkt.futureHorizon)
                #     ti.timeStamp = datetime
                #
                #     #assign marketState property
                #
                #     ti.assign_state(ti.market)
                # ELIMINATE HURKY INLINE CONSTRUCTOR ABOVE
                
                # Store the new TimeInterval in Market.timeIntervals
                
                self.timeIntervals.append(ti)  # = [mkt.timeIntervals, ti]
                
            # The TimeInterval already exists.
            elif tis_len == 1:
                
                # Find the TimeInterval and check its market state assignment.
                #ti = findobj(mkt.timeIntervals, 'startTime', steps(i))
                tis[0].assign_state(self)  # ti.assign_state(mkt)
                
            # Duplicate time intervals exist. Remove all but one.
            else:
                
                # Get rid of duplicate TimeIntervals.
                #mkt.timeIntervals = unique(mkt.timeIntervals)
                #tmp = []
                #for x in mkt.timeIntervals:
                #  if x not in tmp:
                #      tmp.append(x)
                #mkt.timeIntervals = tmp

                # Find the remaining TimeInterval having the startTime step(i).
                #ti = findobj(mkt.timeIntervals, 'startTime', steps(i))
                #tis = [x for x in mkt.timeIntervals if x.startTime == steps[i]]
                self.timeIntervals = [tis[0]]

                # Finish by checking and updating the TimeInterval's
                # market state assignment.
                tis[0].assign_state(self)



    def view_marginal_prices(self, energy_type = MeasurementType.PowerReal):
        import matplotlib.pyplot as plt

        
        # Gather active time series and make sure they are in chronological order
        ti = self.timeIntervals
        ti = [x.startTime for x in ti]  #ti = [ti.startTime]
        ti.sort()  #ti = sort(ti)

        # if there are multiple energy types being handled, find the correct one
        if hasattr(self, 'measurementType'):
            i_energy_type = self.measurementType.index(energy_type)
        else:
            i_energy_type = -1
        
        #if ~isa(mkt, 'Market')
        if not isinstance(self, Market):  # if ~isa(tnm, 'NeighborModel')
            _log.warning('Object must be a NeighborModel or LocalAssetModel')
            return
        else:
            mp = self.marginalPrices

        #mp_ti = [x.timeInterval for x in mp]  #mp_ti = [mp.timeInterval]
        #[~, ind] = sort([mp_ti.startTime])
        #mp = mp(ind)
        #mp = [mp.value]
        if i_energy_type>=0:
            sorted_mp = sorted(mp[i_energy_type], key=lambda x: (x.timeInterval.startTime))
        else:
            sorted_mp = sorted(mp, key=lambda x: (x.timeInterval.startTime))
        mp = [x.value for x in sorted_mp]


        # This can be made prettier as time permits.
        fig = plt.figure()
        ax = plt.axes()

        ax.plot(ti, mp)
        plt.title('Marginal Prices in Active Time Intervals')
        plt.xlabel('time')
        plt.ylabel('marginal price ($/kWh)')

    def view_net_curve(self, i, energy_type=MeasurementType.PowerReal):
        """
        Not completed
        :param i:
        :return:
        """
        import matplotlib.pyplot as plt

        # VIEW_MARGINAL_PRICES() - visualize marginal pricing in active time
        # intervals.
        # mkt - market object

        # Gather active time series and make sure they are in chronological order
        ti = self.timeIntervals
        ti_objs = sort_vertices(ti, 'startTime')
        ti = [x.startTime for x in ti]  # ti = [ti.startTime]
        ti.sort()  # ti = sort(ti)

        # if there are multiple energy types, find the index for this one
        if hasattr(self, 'measurementType') and (energy_type in self.measurementType):
            i_energy_type = self.measurementType.index(energy_type)
        else:
            i_energy_type = -1

        # if ~isa(mkt, 'Market')
        if not isinstance(self, Market):
            _log.warning('Object must be a NeighborModel or LocalAssetModel')
            return
        else:
            mp = self.marginalPrices

        # mp_ti = [x.timeInterval for x in mp]  #mp_ti = [mp.timeInterval]
        # [~, ind] = sort([mp_ti.startTime])
        # mp = mp(ind)
        # mp = [mp.value]
        if i_energy_type>=0:
            sorted_mp = sorted(mp[i_energy_type], key=lambda x: x.timeInterval.startTime)
        else:
            _log.warning('This Market does not transact energy type '+ str(energy_type))
            return
            #sorted_mp = sorted(mp, key=lambda x: x.timeInterval.startTime)
        mp = [x.value for x in sorted_mp]

        # This can be made prettier as time permits.
        fig = plt.figure()
        ax = plt.axes()

        #ax.plot(ti, mp)
        plt.title('Marginal Prices in Active Time Intervals')
        plt.xlabel('time')
        plt.ylabel('marginal price ($/kWh)')

        # Pick out the active time interval that is indexed by input i
        ti_objs = ti_objs[i]
        
        # Find the active system vertices in the indexed time interval
        #vertices = findobj(mkt.activeVertices, 'timeInterval', ti)  # IntervalValues
        if i_energy_type>=0:
            vertices = find_objs_by_ti(self.activeVertices[i_energy_type], ti_objs)
        else:
            vertices = find_objs_by_ti(self.activeVertices, ti_objs)
        
        # Extract the vertices. See struct Vertex.
        vertices = [x.value for x in vertices]  #vertices = [vertices.value]  # active Vertices
        
        # Eliminate any vertices that have infinite marginal price values
        #ind = ~isinf([vertices.marginalPrice])  # an index array
        #vertices = vertices(ind)  # Vertices
        #vertices = [x for x in vertices if x.marginalPrice != float("inf")]
        
        # Sort the active vertices in the indexed time interval by power and by
        # marginal price
        vertices = order_vertices(vertices)  # Vertices
        
        # Calculate the extremes and range of the horizontal marginal-price axis
        x = [x.marginalPrice for x in vertices]
        minx = min(x)  # min([vertices.marginalPrice])  # [$/kWh]
        maxx = max(x)  # max([vertices.marginalPrice])  # [$/kWh]
        xrange = maxx - minx  # [$/kWh]
        if maxx==float('inf') and minx==float('inf'): # in case all marginal prices are inf, still plot to show the power demand
            minx = 0
            xrange = 1
        
        # Calculate the extremes and range of the vertical power axis
        y = [x.power for x in vertices]
        miny = min(y)  # min([vertices.power])  # avg.kW]
        maxy = max(y)  # max([vertices.power])  # avg.kW]
        yrange = maxy - miny  # avg.kW]
        if yrange == 0:
            yrange = 2
        
        # Perform scaling if power range is large
        if yrange > 1000:
            unit = '(MW)'
            factor = 0.001
            miny = factor * miny
            maxy = factor * maxy
            yrange = factor * yrange
        else:
            unit = '(kW)'
            factor = 1.0

        # Start the figure with nicely scaled axes
        #axis([minx - 0.1 * xrange, maxx + 0.1 * xrange, miny - 0.1 * yrange, maxy + 0.1 * yrange])

        
        # Place a marker at each vertex.
        #plot([vertices.marginalPrice], factor * [vertices.power], '*')
        
        # Create a horizontal line at zero.
        #line([minx - 0.1 * xrange, maxx + 0.1 * xrange], [0.0, 0.0], 'LineStyle', '--')
        plt.plot([minx - 0.1 * xrange, maxx + 0.1 * xrange],
                 [0.0, 0.0])

        # Draw a line from the left figure boundary to the first vertex.
        if vertices[0].marginalPrice == float('inf'):
            plt.plot([minx - 0.1 * xrange, 5],
                [factor * vertices[0].power, factor * vertices[0].power])
        else:
            #line([minx - 0.1 * xrange, vertices(1).marginalPrice], [factor * vertices(1).power, factor * vertices(1).power])
            plt.plot([minx - 0.1 * xrange, vertices[0].marginalPrice],
                    [factor * vertices[0].power, factor * vertices[0].power])

            # Draw lines from each vertex to the next. If two successive
            # vertices are not continuous, no line should be drawn.
            for i in range(len(vertices)-1):  #for i = 1:(len(vertices) - 1)
                linestyle = ''
                if vertices[i].continuity == 0 and vertices[i+1].continuity == 0:
                    #line([vertices(i:i + 1).marginalPrice], factor * [vertices(i:i + 1).power], 'LineStyle', 'none')
                    linestyle = ''
                else:
                    #line([vertices(i:i + 1).marginalPrice], factor * [vertices(i:i + 1).power], 'LineStyle', '-')
                    linestyle = '-'

                plt.plot([vertices[i].marginalPrice, vertices[i + 1].marginalPrice],
                        [factor * vertices[i].power, factor * vertices[i + 1].power],
                        linestyle=linestyle)
            
            # Draw a line from the rightmost vertex to the left figure boundary.
            #len = len(vertices)
            #line([vertices(len).marginalPrice, maxx + 0.1 * xrange], [factor * vertices(len).power, factor * vertices(len).power])
            plt.plot([vertices[-1].marginalPrice, maxx + 0.1 * xrange],
                    [factor * vertices[-1].power, factor * vertices[-1].power],
                    linestyle=linestyle)


        # Pretty it up with labels and title
        #ax.plot(ti, mp)
        plt.title('Production Vertices (' + ti_objs.name + ')')
        plt.xlabel('unit price ($/kWh)')
        plt.ylabel('power ' + str(unit))
        plt.show()

