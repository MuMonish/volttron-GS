import numpy as np 
import pandas as pd
import os
#import csv
from datetime import datetime, timedelta, date, time
from thermal_agent_model import ThermalAgentModel
from vertex import Vertex
from auction import Auction
from piecewise_fit import bin_data
from local_asset_model import LocalAssetModel
from measurement_type import MeasurementType
from interval_value import IntervalValue
import helics as h

class Chiller(LocalAssetModel):
    # Chiller Class
    # the chiller interfaces with the thermal_agent_model and the neighbor_model
    # the thermal_agent_model negotiates with the cooling auction
    # the neighbor_model negotiates with for electrical power
    def __init__(self, name = None, size=0.0, energy_types = [MeasurementType.Cooling, MeasurementType.PowerReal]):
        super(Chiller,self).__init__(energy_types=energy_types)
        self.name = name 
        self.size = size # size of system in kW of cooling
        self.min_capacity = 0.0 # minimum operating capacity of chillers
        self.energy_type = ['cooling', 'electricity']
        self.thermalFluid = 'water' #string: this is almost always 'water' for a chiller
        self.vertices = [[] for et in energy_types] # list of vertex class instances defining the efficiency curve
        self.activeVertices = [[] for et in energy_types] #list of Vertex class instances defining the cost vs. power used
        #self.thermalAgentModel = None #thermal agent model object
        #self.neighborModel = None #neighbor agent model object
        self.electrical_use = 0.0 # float indicating the electrical demand to run the chiller
        self.scheduledPowers = [[] for et in energy_types]# float indicating the cooling power setpoint
        self.mass_flowrate = 0.0 # float indicating mass flowrate of cold water through chiller
        self.coefs = [] # fit coefficients for fit curves in format c[0] + c[1]x + c[2]x**2 . . .
        self.coefs_linear = [] # fit coefficients for fit curves in format c[0] + c[1]x + c[2]x**2 . . .
        self.datafilename = None
        self.thermalAuction = None
        self.measurementType = energy_types
        #self.defaultPower = [[]]*len(energy_type)
        self.engagementSchedule = [[] for et in energy_types]
        # fit curves may map to range of temperatures, so they also contain mins and maxs    
    
    def create_default_vertices(self, ti, mkt):
        # create the vertices that define the efficiency curve as a relationship between 
        # electricity and cooling power
        # one vertex should be at the max cooling output setpoint. 
        # one vertex should be at the minimum online output setpoint
        # INPUTS:
        # 
        # OUTPUTS:
        # self.vertices: set of default vertices defining the system generally

        # start by making an efficiency fit curve
        if self.coefs == []:
            self.make_fit_curve()
        coefs = self.coefs
        max_power = self.size
        min_power = self.min_capacity

        # The cost goes to infinity at the upper limit
        # find marginal price at the limit
        max_prod_cost = self.use_fit_curve(self.coefs, max_power, 0.1)
        max_marginal_cost = max_prod_cost/max_power
        # make max vertex
        vertex_max = Vertex(marginal_price=max_marginal_cost, prod_cost=max_prod_cost, power=self.size, continuity=True)
        
        # the power goes to zero at the marginal cost at the lower limit, make (0,0) vertex
        vertex_zero = Vertex(marginal_price=0.0, prod_cost=0.0, power=0.0, continuity=False)
        # find production price at the lower limit
        min_prod_cost = self.use_fit_curve(self.coefs, min_power, 0.1)
        if min_power==0:
            min_marginal_cost = 0.0
        else:
            min_marginal_cost = min_prod_cost/min_power
        vertex_min = Vertex(marginal_price= min_marginal_cost, prod_cost=min_prod_cost, power=min_power)
        # convert vertices to time intervaled values
        
        for t in ti:
            vmax = IntervalValue(self, t, mkt, MeasurementType.ActiveVertex, vertex_max)
            vmin = IntervalValue(self, t, mkt, MeasurementType.ActiveVertex, vertex_min)

            #save values
            self.vertices[0].append(vmax)
            self.vertices[0].append(vmin)
        
        # initialize active vertices
        self.activeVertices = self.vertices
        self.defaultVertices = self.vertices


    def make_fit_curve(self):
        #find the vertices that describe a fit function of the electrical power vs. 
        # cooling supplied data. This function should be run once after object initialization, and again whenever
        # efficiency fit data is updated for the most accurate efficiency fit curve
        #INPUTS:
        # the cooling and electric draw are read off a csv
        #
        #OUTPUTS:
        # coefficients defining the relationship between electricity and cooling are saved
        # to the chiller object

        # ASSUMPTIONS:
        # there is a csv with the same name as the boiler
        # this csv has entries of heat out and fuel consumed

        if self.datafilename == None:
            filename = '/efficiency_curves/'+self.name + '_efficiency.xlsx'
        else:
            filename = self.datafilename
        capacity = []
        cooling = []
        temperature = []
        coefs = []
        size = self.size
        # read the efficiency data, if there is no data file, return None and a warning log
        try: 
            datafile = pd.read_excel(os.getcwd()+filename)
            capacity = datafile['cap']
            efficiency = datafile['cooling']
            # un-normalize the capacity data and remove (0,0) points. We don't want to fit to 0,0, because it is discontinuous
            efficiency = efficiency[capacity!=0]
            capacity = capacity[capacity!=0]
            capacity = size*capacity[efficiency!=0]
            self.min_capacity = min(capacity)
            elec_use = 1/efficiency[efficiency!=0]*(capacity) #make this in ternms of electricity use, not efficiency

            # temperature = temperature[efficiency!=0]
            # #bin the data according to temperature
            # n_bins = 3 # number of bins to separate sample data according to temperature
            # cap_binned, elec_binned, temp_min, temp_max = bin_data(capacity, elec_use, temperature, n_bins)
            # #save limits
            # self.coefs['temp_min'] = temp_min
            # self.coefs['temp_max'] = temp_max
            # make fit curves for each of the binned segments
            regression_order = 4 # fourth order regression should capture curve with high enough accuracy
            # for i in range(n_bins):
            #     cap = cap_binned[i]
            #     elec = elec_binned[i]
            #     coefs_binned = np.flip(np.polyfit(cap, elec, regression_order))
            #     coefs.append(coefs_binned)
            coefs = np.flip(np.polyfit(capacity, elec_use, regression_order),0)
            #remove rounding errors/small stuff
            #coefs[np.abs(coefs)<1e-5] = 0.0
            self.datafilename = filename
        except:
            coefs = [0.0, 3.0] # if there is no data start with an assumed COP of 3
        #save values
        self.coefs = coefs

    def get_vertices_from_linear_model(self, mkt):
        self.activeVertices = {}
        neutral_vertex_e = Vertex(marginal_price=float(mkt.electricity_rate), prod_cost=0.0, power=float(self.size) / self.eff)
        neutral_vertex_h = Vertex(marginal_price=float('inf'), prod_cost=0.0, power=0.0)

        neutral_vertex_c = Vertex(marginal_price=float(mkt.electricity_rate), prod_cost=float(mkt.electricity_rate) * float(self.size) / self.eff, power = float(self.size))
        upper_vertex_c = Vertex(marginal_price=float(mkt.electricity_rate), prod_cost=float(mkt.electricity_rate) * float(self.size) / self.eff, power = float(self.size))
        lower_vertex_c = Vertex(marginal_price=float(mkt.electricity_rate), prod_cost=0.0, power=0.0)

        vertices_val = [neutral_vertex_e, neutral_vertex_h, [neutral_vertex_c, lower_vertex_c, upper_vertex_c]]
        vertices_type = [MeasurementType.PowerReal, MeasurementType.Heat, MeasurementType.Cooling]
        mkt_time = mkt.marketClearingTime
        for type_energy, vert in enumerate(vertices_val):
            iv = IntervalValue(self, mkt_time, mkt, MeasurementType.ActiveVertex, vert)
            if str(vertices_type[type_energy]) in self.activeVertices:
                self.activeVertices[str(vertices_type[type_energy])].append(iv)
            else:
                self.activeVertices[str(vertices_type[type_energy])] = []
                self.activeVertices[str(vertices_type[type_energy])].append(iv)

    def update_dispatch(self, mkt, fed = None, helics_flag = bool(0)):

        cool_dispatched = self.scheduledPowers[str(MeasurementType.Cooling)]
        elec_consumed = self.use_fit_curve(cool_dispatched)
        cost = mkt.electricity_rate * elec_consumed

        if helics_flag == True:
            key1 = "WSU_C_GLD_" + self.name + "_power_A"
            key2 = "WSU_C_GLD_" + self.name + "_power_B"
            key3 = "WSU_C_GLD_" + self.name + "_power_C"
            try:
                pubA = h.helicsFederateGetPublication(fed, key1)
                pubB = h.helicsFederateGetPublication(fed, key2)
                pubC = h.helicsFederateGetPublication(fed, key3)
                status = h.helicsPublicationPublishComplex(pubA, elec_consumed*1000/3, 0)
                status = h.helicsPublicationPublishComplex(pubB, elec_consumed*1000/3, 0)
                status = h.helicsPublicationPublishComplex(pubC, elec_consumed*1000/3, 0)
                print('Data {} Published to GLD {} via Helics -->'.format(elec_consumed*1000, self.name))
            except:
                print('Publication was not registered')

        interval = mkt.marketClearingTime.strftime('%Y%m%dT%H%M%S')
        line_new = str(mkt.marketClearingTime) + "," + str(interval) + "," + str(cool_dispatched) + "," + str(elec_consumed) + "," + str(cost) + " \n"
        file_name = os.getcwd() + '/Outputs/' + self.name + '_output.csv'
        try:
            with open(file_name, 'a') as f:
                f.writelines(line_new)
        except:
                f = open(file_name, "w")
                f.writelines("TimeStamp,TimeInterval,Cool Dispatched,Electricity Consumed,Cost\n")
                f.writelines(line_new)
        f.close()


    def use_fit_curve(self, setpoint):
        # find the fuel use for the given power setting
        # INPUTS: coefficients for electricity vs. cooling power
        # setpoint: power setpoint in kW
        # other_asset_price: cost of electricity in units compatible with fit curve
        #
        # OUTPUTS: cost at power setpoint in [$/kWh]
        coefs = self.coefs
        cost = 0
        for i in range(len(coefs)):
            cost = cost + coefs[i]*setpoint**(i)
        #cost = cost*other_asset_price
        cost = max(cost,0)
        return cost

    def update_active_vertex(self, Csetpoint, Tamb, e_market_price, auc):
        # find the electrical and cooling vertices that are active given the cooling setpoints
        # INPUTS:
        # - Csetpoint: setpoint from auction
        # - Tamb: ambient temperature in degrees C
        # - e_market_price: electricity market price from neighbor model
        # - auc.marginal_prices: cooling market marginal price from auction
        #
        # OUTPUTS:
        # - activeVertices_e: list of active vertices for this dispatch associated with the
        #       electrical market
        # - activeVertices_c: list of active vertices for this dispatch associated
        #       with the cooling auction

        #read in inputs:
        c_market_price = auc.marginal_prices

        # update the agent's values
        self.scheduledPowers = [Csetpoint]
        self.find_electric_use(self,Tamb)
        self.find_massflow(self)
        coefs = self.coefs
        
        #use those values to create new vertices        
        # create an intermediate vertex at the setpoint
        cooling_cost = self.use_fit_curve(coefs=coefs, setpoint=Csetpoint, other_asset_price=e_market_price)
        Esetpoint = cooling_cost/e_market_price
        electric_cost = Csetpoint*c_market_price
        # find the price for one more kWh of cooling
        cooling_cost_upper = self.use_fit_curve(coefs=coefs, setpoint=Csetpoint+1, other_asset_price=e_market_price)
        marginal_price_cooling = cooling_cost_upper-cooling_cost

        # create vertices
        nominal_vertex_c = Vertex(marginal_price=marginal_price_cooling, prod_cost=cooling_cost, power=Csetpoint)
        nominal_vertex_e = Vertex(marginal_price=c_market_price, prod_cost=electric_cost, power=Esetpoint)        

        # update agent state
        self.activeVertices_c = [nominal_vertex_c, self.vertices[0], self.vertices[1]]
        self.activeVertices_e = [nominal_vertex_e, self.vertices[0], self.vertices[1]]

    def find_electric_use(self, Tamb):
        # find the electrical use given the cooling setpoint and ambient temperature
        # INPUTS:
        # - Csetpoint: fload for setpoint from auction
        # - coefs: list of fit curve coefficients
        # - Tamb: ambient temperature
        #
        # OUTPUTS:
        # - electrical_use: float amount of electrical energy used to power the chiller to meet the cooling setpoint
         
        Csetpoint = self.scheduledPowers[0]
        electrical_use = 0.0
        # find bin associated with ambient temperature
        bin_min = self.coefs['temp_min']
        bin_max = self.coefs['temp_max']
        bin_index = 0
        # find index of the bin where Tamb is in that bin
        for i in range(len(bin_min)):
            if Tamb>=bin_min[i] and Tamb<= bin_max[i]:
                bin_index = i
        # find electrical power draw from chiller to produce the cooling power setpoint with the given ambients
        if Csetpoint>0:
            coefs = self.coefs[bin_index]
            for i in range(len(coefs)):
                electrical_use = electrical_use + coefs[i]*Csetpoint**(i)
        # save it
        self.electrical_use = electrical_use

    def find_massflow(self, auc):
        # find the cold water mass flowrate through the chiller given the cooling power provided and the
        # supply and return water temperature setpoints
        # INPUTS:
        # - chiller object which contains power setpoint
        # - auc: cold water auction object which contains water temperature values
        # 
        # OUTPUTS:
        # - massflow: mass flowrate of cold water through the chiller
        
        mass_flowrate = 0.0
        Treturn = auc.Treturn
        Tsupply = auc.Tsupply
        Csetpoint = self.scheduledPowers
        # find specific heat of water at 4 C (assumed cold water temperature)
        Cp = 4.2032 # kJ/kg K assume pipes are not pressurized (1 atm) and T return = 4C
        # if the chiller is on, calculate it, otherwise the mass_flowrate is 0
        if Csetpoint>0:
            mass_flowrate = Csetpoint/(Cp*(Treturn-Tsupply))
        # save value
        self.mass_flowrate = mass_flowrate
