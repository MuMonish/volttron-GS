import numpy as np 
import pandas as pd
from datetime import datetime, timedelta, date
import os
import csv
from thermal_agent_model import ThermalAgentModel 
from measurement_type import MeasurementType
from vertex import Vertex
from auction import Auction
from local_asset_model import *
from transactive_record import TransactiveRecord

import helics as h
import random
import logging
import json
import time

class FlexibleBuilding(LocalAssetModel):
    # Building with flexible loads
    # the electrical loads interface with the electrical node and are not typically flexible
    # the thermal loads interface with the thermal auctions and are flexible
    # a change in thermal load may affect the electrical loads
    def __init__(self, energy_types = [MeasurementType.PowerReal, MeasurementType.Heat, MeasurementType.Cooling]):
        super(FlexibleBuilding, self).__init__(energy_types=energy_types)
        self.cost_curve = {} # fit curve parameters for cost associated with load deviation
        self.cost_of_deviation = 0.0 # cost of allowing temperature to deviate
        self.datestamp = datetime(2010, 1, 1, 0, 0, 0)
        self.historicalProfile = None # historical load profiles from csv
        self.internal_mass = 1# mCp portion of the mCp(T1-T0) equation representing the internall mass of the building
        self.loadForecast = [[0.0] for et in energy_types] # default electircal load profile before flexibility [kW avg.]
        self.mass_flowrate = [None, 0.0, 0.0] # mass flowrate of thermal fluids in same order
        self.name = None 
        self.Tactual = 23 # actual temperature in building in degrees C
        self.thermalAuction = None
        self.thermalFluid = [None,'steam', 'water'] # thermal fluids associated with each energy transfer above
        self.Tlb = 21 # temperature lower limit before incurring extreme deviation costs
        self.Tset = 23 # temperature setpoint for building in degrees C, based on historical profile
        self.Tub = 25 # temperature upper limit before incurring extreme deviation costs
        self.vertices = [] # vertices describing marginal cost pricing
        self.receivedSignal = []

    def find_deviation_cost(self, Tactual=None, find_limit=False, ti=None):
        #make the curves that describe the cost of deviating from the original load
        # this function should be run once after object initiation and again
        # any time the loads from the historical profile ("historicLoad") are updated
        # INPUTS:
        # loadForecast_e: load profile created from historic data and datestamp
        #       this is the electric load if there were no flexibility
        # loadForecast_h: same as above, but for heat
        # loadForecast_c: same as above, but for cooling
        # Tset: same as above, but for building temperature setpoint
        #
        # OUTPUTS:
        # cost_curve: coefficients describing the cost of deviating from the 
        #       desired load. This is based on temperature deviation for thermal 
        #       loads, and on time of day for electricity laod
        #
        # ASSUMPTIONS:
        # The forecasted load is the power required to maintain the actual temperature
        # So any deviation in the load results in deviation of the building temperature
        #
        # the cost equation is in the form: 
        # c((Tactual-Tset)/(Tub-Tset))^g if Tzone>=Tset 
        # c((Tactual-Tset)/(Tlb-Tset))^g if Tzone<Tset
        c = 1
        g = 2
        # the same c[0] and c[1] are used for both equations, because
        # the cost is symmetric. Being too hot is considered equally harmful as being too cold
        #update inputs:
        if ti == None:
            ti = self.datestamp

        self.update_load_forecast(ti)
        Tset = self.Tset
        Tub = self.Tub
        Tlb = self.Tlb
        #Tactual = self.Tactual
        loade = self.loadForecast[0]
        loadh = self.loadForecast[1]
        loadc = self.loadForecast[2]

        # if there is no input Tactual, then you are just trying to find the marginal cost
        if Tactual==None:
            extra_heat = 1 #kWh of heat
            extra_cooling = 1# kWh of cooling
            #calculate the temperature if you had one less kWh of heat
            Tactual_h = - extra_heat/self.internal_mass + Tset
            Tactual_c = extra_cooling/self.internal_mass + Tset
        # you supplied an actual temperature of the building use that temp
        else: 
            Tactual_h = Tactual
            Tactual_c = Tactual
            # first figure out if you need more heating and cooling just to get to the setpoint
            # this will occur if you deviated in the last timestep and so are no longer at the setpoint
            extra_heat = self.internal_mass * (Tset-Tactual) #[kWs to kWh]
            extra_cooling = self.internal_mass * (Tactual-Tset) #[kWs to kWh]

        T_deviation_c = Tactual_c-Tset
        T_deviation_h = Tset-Tactual_h
        
        # add the extra load value to the forecasted load which is more prominent. For example,
        # if the cooling load is larger, the building is in "cooling mode" so add the excess thermal
        # value to the forecasted cooling load
        # only add to one of them to avoid over counting
        if loadh > loadc:
            loadh = loadh + extra_heat
        else:
            loadc = loadc + extra_cooling

        # find effects on electrical load from changing heat and cooling loads
        #self.find_electrical_load_effects(loadh, loadc)

        # calculate the cost of deviation from this new load
        # since formula is symetrical, just calculate in one direction
        deviation_cost_c = c*(T_deviation_c/(Tub-Tset))**g 
        deviation_cost_h = c*(T_deviation_h/(Tset-Tlb))**g
        # record deviation costs and updated loads
        # if you are just checking for marginal prices, only return marginal prices
        if Tactual==None:
            return deviation_cost_c, deviation_cost_h
        # if you are calculating values after convergence, save those
        elif find_limit==True:
            return deviation_cost_c, deviation_cost_h
        else:
            #self.deviation_load_h = extra_heat
            #self.deviation_load_c = extra_cooling
            self.scheduledPowers[1] = loadh
            self.scheduledPowers[2] = loadc
            self.cost_of_deviation = deviation_cost_c+deviation_cost_h

    def create_default_vertices(self, ti, mkt):
        # creaate vertices that are use on instantiation and whenever communication is lost
        # INPUTS:
        # 
        # OUTPUTS:
        # vertices: the default minimum and maximum limit vertices
        for t in ti:
            self.update_active_vertex(t, mkt)
        self.vertices = [self.activeVertices[0], self.activeVertices[1], self.activeVertices[2]]
        self.defaultVertices = [[self.activeVertices[0][0]], [self.activeVertices[1][0]], [self.activeVertices[2][0]]]
        self.defaultPower = [self.activeVertices[0][0].value.power, self.activeVertices[1][0].value.power, self.activeVertices[2][0].value.power]

    ##  This function is used to extract vertices from the CESI building models developed at WSU
    def get_vertices_from_CESI_building(self, mkt):
        # creaate vertices that are use on instantiation and whenever communication is lost
        # INPUTS:
        #
        # OUTPUTS:
        # vertices: the default minimum and maximum limit vertices
        Signal = {}
        filename = "flexible_building_data.txt"  # this loop will need to be changed if multiple buildings
        with open(filename) as file:
            reader = csv.DictReader(file)

            # Extract the interval information into transactive records.
            # NOTE: A TransactiveRecord constructor is being used.
            for row in reader:  # for i = 1:r
                if 'E_Type' in row:
                    transactive_record = TransactiveRecord(ti=row['TimeInterval'],
                                                      rn=float(row['Record']),
                                                      mp=float(row['MarginalPrice']),
                                                      p=float(row['Power']),
                                                      pu=float(row['PowerUncertainty']),
                                                      cost=float(row['Cost']),
                                                      rp=float(row['ReactivePower']),
                                                      rpu=float(row['ReactivePowerUncertainty']),
                                                      v=float(row['Voltage']),
                                                      vu=float(row['VoltageUncertainty']),
                                                      e_type=int(row['E_Type']))
                else:
                    transactive_record = TransactiveRecord(ti=row['TimeInterval'],
                                                        rn=float(row['Record']),
                                                        mp=float(row['MarginalPrice']),
                                                        p=float(row['Power']),
                                                        pu=float(row['PowerUncertainty']),
                                                        cost=float(row['Cost']),
                                                        rp=float(row['ReactivePower']),
                                                        rpu=float(row['ReactivePowerUncertainty']),
                                                        v=float(row['Voltage']),
                                                        vu=float(row['VoltageUncertainty']))

                # Save each transactive record (NOTE: We can apply more savvy to find
                # and replace the signal later.)
                #self.receivedSignal = [self.receivedSignal, transative_record]
                if row['TimeInterval'] in Signal:
                    Signal[row['TimeInterval']].append(transactive_record)
                else:
                    Signal[row['TimeInterval']] =[]
                    Signal[row['TimeInterval']].append(transactive_record)
        self.create_vertices_from_CESI_building_record(Signal, mkt)

    ##  This function is used to extract vertices from the CESI building models records_developed at WSU
    def request_helics_to_get_vertices_from_CESI_building(self, mkt, fed):
        # FUNCTION RECEIVE_TRANASCTIVE_SIGNAL USING HELICS () - receive and save transactive
        # FUNCTION RECEIVE_TRANASCTIVE_SIGNAL() - receive and save transactive
        # records from a transactive Neighbor object.
        # (NOTE: In the Matlab implementation, the transactive signals are
        # "received" via a readable csv file.)
        # mtn = myTransactiveNode object
        # obj - the NeighborModel object
        #
        # The process of receiving a transactive signal is emulated by reading an
        # available text table that is presumed to have been created by the
        # transactive neighbor. This process may change in field settings and using
        # Python and other code environments.

        # If trying to receive a transactive signal from a non-transactive neighbor,
        # create a warning and return.
        # if not self.transactive:
        #     _log.warning('Transactive signals are not expected to be received from non-transactive neighbors. '
        #                  'No signal is read.')
        #     return

        # Here is the format for the preferred text filename. (NOTE: The name is
        # applied by the transactive neighbor and is not under the direct control
        # of myTransactiveNode.)
        # The filename starts with a shortened name of the originating node.
        source_node = str(self.name)
        if len(source_node) > 5:
            source_node = source_node[0:5]  # source_node(1:5)

        # Shorten the name of the target node
        target_node = str(mkt.name)
        if len(target_node) > 5:
            target_node = target_node[0:5]  # target_node(1:5)

        # Format the filename. Do not allow spaces.
        Signal = {}
        filename = ''.join([source_node, '-', target_node, '.txt'])
        filename = filename.replace(' ', '')
        sub_key = source_node+'_'+target_node
        #filename = 'Python/'+filename

        t = mkt.marketClearingTime.hour*3600

        grantedtime = -1
        while grantedtime < t:
            grantedtime = h.helicsFederateRequestTime(fed, t)
        time.sleep(0.1)

        #######################   Subscribing   ############################
        print(sub_key)
        sub = h.helicsFederateGetSubscription(fed, sub_key)
        sub_data = h.helicsInputGetString(sub)
        print('Time requested via Helics -->', t)
        print('Data obtained via Helics -->', sub_data)
        print('Already Received Data @ ', grantedtime)

            # Extract the interval information into transactive records.
            # NOTE: A TransactiveRecord constructor is being used.
        for row in csv.DictReader(sub_data.splitlines()):  # for i = 1:r
            if 'E_Type' in row:
                transactive_record = TransactiveRecord(ti=row['TimeInterval'],
                                                      rn=int(row['Record']),
                                                      mp=float(row['MarginalPrice']),
                                                      p=float(row['Power']),
                                                      pu=float(row['PowerUncertainty']),
                                                      cost=float(row['Cost']),
                                                      rp=float(row['ReactivePower']),
                                                      rpu=float(row['ReactivePowerUncertainty']),
                                                      v=float(row['Voltage']),
                                                      vu=float(row['VoltageUncertainty']),
                                                      e_type=int(row['E_Type']))
            else:
                transactive_record = TransactiveRecord(ti=row['TimeInterval'],
                                                        rn=int(row['Record']),
                                                        mp=float(row['MarginalPrice']),
                                                        p=float(row['Power']),
                                                        pu=float(row['PowerUncertainty']),
                                                        cost=float(row['Cost']),
                                                        rp=float(row['ReactivePower']),
                                                        rpu=float(row['ReactivePowerUncertainty']),
                                                        v=float(row['Voltage']),
                                                        vu=float(row['VoltageUncertainty']))

                # Save each transactive record (NOTE: We can apply more savvy to find
                # and replace the signal later.)
                #self.receivedSignal = [self.receivedSignal, transative_record]
            if row['TimeInterval'] in Signal:
                Signal[row['TimeInterval']].append(transactive_record)
            else:
                Signal[row['TimeInterval']] = []
                Signal[row['TimeInterval']].append(transactive_record)
        self.create_vertices_from_CESI_building_record(Signal, mkt)

    def create_vertices_from_CESI_building_record(self, Signal, mkt):
        self.activeVertices={}
        for interval_time in Signal:
            for row in Signal[interval_time]:
                if row.e_type == MeasurementType.PowerReal:
                    neutral_vertex_e = Vertex(marginal_price=float('inf'), prod_cost=row.cost, power=row.power)
                elif row.e_type == MeasurementType.Heat:
                    if row.record == 0:
                        neutral_vertex_h = Vertex(marginal_price=row.marginalPrice, prod_cost=row.cost, power=row.power)
                    elif row.record == 1:
                        lower_vertex_h = Vertex(marginal_price=row.marginalPrice, prod_cost=row.cost, power=row.power)
                    elif row.record == 2:
                        upper_vertex_h = Vertex(marginal_price=row.marginalPrice, prod_cost=row.cost, power=row.power)
                elif row.e_type == MeasurementType.Cooling:
                    if row.record == 0:
                        neutral_vertex_c = Vertex(marginal_price=row.marginalPrice, prod_cost=row.cost, power=row.power)
                    elif row.record == 1:
                        lower_vertex_c = Vertex(marginal_price=row.marginalPrice, prod_cost=row.cost, power=row.power)
                    elif row.record == 2:
                        upper_vertex_c = Vertex(marginal_price=row.marginalPrice, prod_cost=row.cost, power=row.power)
            vertices_val = [[neutral_vertex_e], [neutral_vertex_h, lower_vertex_h, upper_vertex_h], [neutral_vertex_c, lower_vertex_c, upper_vertex_c]]
            vertices_type = [MeasurementType.PowerReal, MeasurementType.Heat, MeasurementType.Cooling]

            for type_energy, vert in enumerate(vertices_val):
                iv = IntervalValue(self, interval_time, mkt, MeasurementType.ActiveVertex, vert)
                if str(vertices_type[type_energy]) in self.activeVertices:
                    self.activeVertices[str(vertices_type[type_energy])].append(iv)
                else:
                    self.activeVertices[str(vertices_type[type_energy])] =[]
                    self.activeVertices[str(vertices_type[type_energy])].append(iv)

    def update_dispatch(self, mkt, fed, helics_flag = bool(1)):
        for i in range(len(self.measurementType)):
            if self.measurementType[i] == MeasurementType.PowerReal:
                elec_dispatched = self.scheduledPowers[i]
            elif self.measurementType[i] == MeasurementType.Heat:
                heat_dispatched = self.scheduledPowers[i]
            elif self.measurementType[i] == MeasurementType.Cooling:
                cool_dispatched = self.scheduledPowers[i]

        interval = mkt.marketClearingTime.strftime('%Y%m%dT%H%M%S')


        header = str("TimeStamp,TimeInterval,Temperature Setpoint Dispatched,Heat Dispatched,Cooling Dispatched\n")
        new_line = str(mkt.marketClearingTime) + "," + str(interval) + "," + str(elec_dispatched) +  "," + str(heat_dispatched) + "," + str(cool_dispatched) + " \n"

        t = mkt.marketClearingTime.hour * 3600
        t = t+1

        if helics_flag == True:
            target_node = str(self.name)
            if len(target_node) > 5:
                target_node = target_node[0:5]  # source_node(1:5)

            # Shorten the name of the target node
            source_node = str(mkt.name)
            if len(source_node) > 5:
                source_node = source_node[0:5]  # target_node(1:5)

            # Format the filename. Do not allow spaces.
            pub_key = source_node+'_'+target_node
            pub_data = header + new_line

            grantedtime = -1
            while grantedtime < t:
                grantedtime = h.helicsFederateRequestTime (fed, t)
            time.sleep(0.1)

            pub = h.helicsFederateGetPublication(fed, pub_key)
            print(pub)
            status = h.helicsPublicationPublishString(pub, pub_data)
            print('Data Published to Helics -->', pub_data)

            print('Time requested via Helics -->', t)
            print('Time granted via Helics -->', grantedtime)

        file_name = os.getcwd() + '/Outputs/' + self.name + '_output.csv'
        try:
            with open(file_name, 'a') as f:
                f.writelines(new_line)
        except:
                f = open(file_name, "w")
                f.writelines(header)
                f.writelines(new_line)
        f.close()



    def update_active_vertex(self, ti, mkt):
        # update active vertices based on the load forecast and the temperature setpoint
        # INTPUTS:
        # Tset: temperature setpoint
        # deviation_cost: cost of deviation from load by 1 kWh
        # 
        # OUTPUTS:
        # self.activeVertices_e: active vertices associated with electrical load
        # self.activeVertices_h: active vertices associated with heat load
        # self.activeVertices_c: active vertices associated with cooling load

        # read in agent's properties
        internal_mass = self.internal_mass
        Tset = self.Tset
        Tub = self.Tub
        Tlb = self.Tlb
        # update the agent's values
        self.update_load_forecast(ti)
        neutral_load_e = self.loadForecast[0]
        neutral_load_h = self.loadForecast[1]
        neutral_load_c = self.loadForecast[2]
        deviation_cost_c, deviation_cost_h = self.find_deviation_cost(ti=ti)
        # if setpoints haven't been established yet, assume they are the neutral load
        if self.scheduledPowers[0] == None:
            Esetpoint = neutral_load_e
            Hsetpoint = neutral_load_h
            Csetpoint = neutral_load_c
        else:
            Esetpoint = self.scheduledPowers[0]
            Hsetpoint = self.scheduledPowers[1]
            Csetpoint = self.scheduledPowers[2]

        # if this is the firt time through, the setpoints will be zero
        if Esetpoint == []:
            Esetpoint = 0.0
        if Hsetpoint == []:
            Hsetpoint = 0.0
        if Csetpoint == []:
            Csetpoint = 0.0

        #deviation_heat = self.deviation_load_h # the deviation in the heat load if you went to the min temp setpoint
        #deviation_cool = self.deviation_load_c # the deviation in cooling load if you went to the max temp setpoint
        self.find_massflow_steam(Hsetpoint)
        self.find_massflow_water(Csetpoint)
        
        #use values to create new vertices

        # flexible buildings create atleast three vertices:one at neutral, one at the hot side of neutral, and one at the cold side
        neutral_vertex_e = Vertex(marginal_price=float('inf'), prod_cost=0.0, power=-neutral_load_e) # there is no flexibility in electrical
        # the marginal price from the neutral point is the deviation cost
        neutral_vertex_h = Vertex(marginal_price=deviation_cost_h/neutral_load_h, prod_cost=0.0, power = -neutral_load_h)
        neutral_vertex_c = Vertex(marginal_price=deviation_cost_c/neutral_load_c, prod_cost=0.0, power = -neutral_load_c)

        # make vertices at upper and lower limits
        # find power at upper limit
        upper_power_h = neutral_load_h + (Tub-Tset)*internal_mass
        upper_power_c = neutral_load_c + (Tset-Tlb)*internal_mass
        # find power at lower limit
        lower_power_h = max(neutral_load_h - (Tset-Tlb)*internal_mass, 0)
        lower_power_c = max(neutral_load_c - (Tub-Tset)*internal_mass, 0)
        #find costs of max deviation
        lower_cost_c, upper_cost_h = self.find_deviation_cost(Tactual=Tub, find_limit=True, ti = ti)
        upper_cost_c, lower_cost_h = self.find_deviation_cost(Tactual=Tlb, find_limit=True, ti = ti)
        # find marginal cost
        if lower_power_h==0:
            marginal_price_lower_h = lower_cost_h
        else:
            marginal_price_lower_h = lower_cost_h/lower_power_h
        if lower_power_c==0:
            marginal_price_lower_c = lower_cost_c
        else:
            marginal_price_lower_c = lower_cost_c/lower_power_c
        #make verticies
        upper_vertex_h = Vertex(marginal_price=0, prod_cost= upper_cost_h, power = -upper_power_h)#Vertex(marginal_price=upper_cost_h/upper_power_h, prod_cost= upper_cost_h, power = -upper_power_h)
        upper_vertex_c = Vertex(marginal_price=0, prod_cost= upper_cost_c, power = -upper_power_c)#Vertex(marginal_price=upper_cost_c/upper_power_c, prod_cost= upper_cost_c, power = -upper_power_c)
        lower_vertex_h = Vertex(marginal_price=float('inf'), prod_cost= lower_cost_h, power = -lower_power_h)#Vertex(marginal_price=marginal_price_lower_h, prod_cost= lower_cost_h, power = -lower_power_h)
        lower_vertex_c = Vertex(marginal_price=float('inf'), prod_cost= lower_cost_c, power = -lower_power_c)#Vertex(marginal_price=marginal_price_lower_c, prod_cost= lower_cost_c, power = -lower_power_c)

        vertices_val = [[neutral_vertex_e],[neutral_vertex_h, lower_vertex_h, upper_vertex_h], [neutral_vertex_c, lower_vertex_c, upper_vertex_c]]

        for my_energy_type in range(len(self.measurementType)):

            iv = find_obj_by_ti(self.activeVertices[my_energy_type], ti)
            # If the active vertex does not exist, a new interval value must be
            # created and stored.
            if iv is None:
                for vert in vertices_val[my_energy_type]:
                    # Create the interval value and place the active vertex in it
                    iv = IntervalValue(self, ti, mkt, MeasurementType.ActiveVertex, vert)
                    # Append the interval value to the list of active vertices
                    self.activeVertices[my_energy_type].append(iv)
            else:
                # Otherwise, simply reassign the active vertex value to the
                iv.value = vertices_val[my_energy_type]


    def find_electrical_load_effects(self, new_loadh, new_loadc):
        # find the new electrical load given the flexibility in thermal loads
        pass


    def update_load_forecast(self, ti):
        # find the historical load profiles associated with today to predict today's loads
        # 
        # INPUTS:
        # - historicalProfile: load profiles with date and temperature stamps to used in forecast
        # - datestamp: list of date/times for this horizon 
        #
        # OUTPUTS:
        # - loadForecast_e: electric load profile created from historical data forecast
        # - loadForecast_h: heat load profile created from historical data forecast
        # - loadForecast_c: cooling load profile created from historical data forecast
        #
        # ASSUMPTIONS:
        # there is a csv with the same name as the flexibile_building object
        # which has historical load data in the format: 
        # date, Temperature, electric load, heat load, cooling load

        datestamp = ti.timeStamp.toordinal()-365*10-2#self.datestamp

        # load historical data if you are on the first timestep
        if self.historicalProfile == None:
            try:
                filename = self.name + '.xlsx'
                datafile = pd.read_excel(os.getcwd()+filename)
            except:
                datafile = pd.read_excel(os.getcwd()+'/test_data/wsu_campus_2009_2012.xlsx')
            e_load = datafile[self.name+'_E']
            h_load = datafile[self.name+'_H']
            c_load = datafile[self.name+'_C']
            timestamp = datafile['timestamp']
            hist_profile = {}
            hist_profile['e_load'] = e_load
            hist_profile['h_load'] = h_load
            hist_profile['c_load'] = c_load
            hist_profile['timestamp'] = timestamp-366#[datetime.fromordinal(int(ts)-366) + timedelta(days = ts%1) for ts in timestamp]
            self.historicalProfile = hist_profile
        
        # need to interpolate if dates and times don't exactly line up
        # datestamp = datestamp.toordinal()
        if 'e_load' in self.historicalProfile:
            loadForecast_e = np.interp(datestamp, self.historicalProfile['timestamp'], self.historicalProfile['e_load'])
            self.loadForecast[0] = loadForecast_e   
        if 'h_load' in self.historicalProfile:
            loadForecast_h = np.interp(datestamp, self.historicalProfile['timestamp'], self.historicalProfile['h_load'])
            self.loadForecast[1] = loadForecast_h
        if 'c_load' in self.historicalProfile:
            loadForecast_c = np.interp(datestamp, self.historicalProfile['timestamp'], self.historicalProfile['c_load'])
            self.loadForecast[2] = loadForecast_c
        if 'Tset' in self.historicalProfile:
            Tset = np.interp(datestamp, self.historicalProfile['timestamp'], self.historicalProfile['Tset'])
            self.Tset = Tset
        
        # may want to do more advanced forecasting later
   

    def find_massflow_steam(self, Hsetpoint):
        # find the massflow of steam required to meet the heat load
        # INPUTS:
        # auc: heat auction object, this object contains:
        # - auc.Tsupply: temperature of steam supply loop from auction
        # - auc.Treturn: temperature of steam return loop from auction
        # Hsetpoint: heat generated setpoint in kW (this value is negative because it is a load)
        # 
        # OUTPUTS:
        # - self.mass_flowrate: this list contains the mass flowrates for electric (None), heat (steam), and cooling (water)

        # pull values
        auc = self.thermalAuction[0]
        Tsupply = auc.Tsupply
        Treturn = auc.Treturn
        if Hsetpoint == []:
            Hsetpoint = 0
        # find the specific heat of steam return temperature
        Cp = 2.014 #[kJ/kgK]
        # calculate mass flowrate
        mfr = Hsetpoint/(Cp*(Tsupply-Treturn)) # [kg/s]
        self.mass_flowrate[1] = mfr     

    def find_massflow_water(self, Csetpoint):
        # find the massflow of water required to meet the cooling load
        # INPUTS:
        # auc: cooling auction object, this object contains:
        # - auc.Tsupply: temperature of cold water supply loop from auction
        # - auc.Treturn: temperature of cold water return loop from auction
        # Csetpoint: cooling generated setpoint in kW (this value is negative because it is a load)
        #
        # OUTPUTS:
        # - self.mass_flowrate: this list contains the mass flowrates for electric (None), heat (steam), and cooling (water)

        # pull values
        auc = self.thermalAuction[1]
        Tsupply = auc.Tsupply
        Treturn = auc.Treturn

        if Csetpoint == []:
            Csetpoint = 0
        # find the specific heat of water return temperature
        Cp = 4.2032 #[kJ/kgK] assume pipes are not pressurized (1 atm, 4C)
        # calculate massflow
        mfr = Csetpoint/(Cp*(Treturn-Tsupply))
        # save value
        self.mass_flowrate[2] = mfr 