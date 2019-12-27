#################################
#### MyTransactiveNode with thermal Test
from datetime import timedelta, datetime

from neighbor import Neighbor
from neighbor_model import NeighborModel
from local_asset import LocalAsset
from myTransactiveNode import myTransactiveNode
from meter_point import MeterPoint
from measurement_type import MeasurementType
from measurement_unit import MeasurementUnit
from temperature_forecast_model import TemperatureForecastModel
#from market import Market
from market_WSU import Market
from market_state import MarketState
from auction import Auction
from flexible_building import FlexibleBuilding
from inflexible_building import InflexibleBuilding
from vertex import Vertex
from helpers import prod_cost_from_vertices
from interval_value import IntervalValue
from gas_turbine import GasTurbine
from boiler import Boiler
from chiller import Chiller
import copy

from helics_functions import create_config_for_helics
from helics_functions import create_broker
from helics_functions import register_federate
from helics_functions import destroy_federate
import helics as h


# create a neighbor model
WSU_Campus = myTransactiveNode()
mTN = WSU_Campus
mTN.description = 'WSU_Campus'
mTN.name = 'WSU_Campus'

#set up AVISTA power meter
SCUE_meter = MeterPoint()
MP = SCUE_meter
MP.description = 'meters SCUE building electric use from AVISTA'
MP.measurementType = MeasurementType.PowerReal
MP.measurement = MeasurementUnit.kWh
SCUE_meter = MP

# provide a cell array of all the MeterPoints to myTransactiveNode
mTN.meterpoints = [SCUE_meter]

## instantiate each Information Service Model
# this is a service that can be queried for information
# includes model prediction for future time intervals
# Pullman Temperature Forecast <-- Information service model
PullmanTemperatureForecast = TemperatureForecastModel()
ISM = PullmanTemperatureForecast
ISM.name = 'PullmanTemperatureForecast'
ISM.predictedValues = [] # dynamically assigned

mTN.informationServiceModels = [PullmanTemperatureForecast]

########################################################################################################
## Instantiate Markets and Auctions
# Markets specify active TimeIntervals
# Auctions handle thermal loads

## Day Ahead Market
date_string = '2017-01-18'
dayAhead = Market(measurementType = [MeasurementType.PowerReal, MeasurementType.Heat, MeasurementType.Cooling])
MKT = dayAhead
MKT.name = 'WSU_Campus_Market'
MKT.commitment = False # start without having commited any resources
MKT.converged = False # start without having converged
MKT.defaultPrice = [0.03, 0.01, 0.02] # [$/kWh]
MKT.electricity_rate = 0.0551
MKT.gas_rate = 5.6173/293.07
MKT.diesel_rate = 24.0/12.5
MKT.dualityGapThreshold = 0.001 #optimal convergence within 0.1Wh
MKT.futureHorizon = timedelta(hours=24)
MKT.intervalDuration = timedelta(hours=1)
MKT.intervalsToClear = 24 # clear entire horizon at once, this is a default value
MKT.marketClearingTime = datetime.strptime(date_string, "%Y-%m-%d").replace(hour=0, minute=0, second=0, microsecond=0) # align with top of hour
MKT.marketOrder = 1 # this is the first and only market
MKT.nextMarketClearingTime = MKT.marketClearingTime + timedelta(hours=0)
MKT.initialMarketState = MarketState.Inactive
dayAhead = MKT
dayAhead.check_intervals()

ti = dayAhead.timeIntervals
mTN.markets = [dayAhead]#, cooling_auction, heat_auction]

###################################################################################datetime.strptime('2017-01-18', "%Y-%m-%d")#####################

## Instantiate Neighbors and NeighborModels
Centralized_Dispatcher = Neighbor()
NB = Centralized_Dispatcher
NB.lossFactor = 0.01 # one percent loss at full power (only 99% is seen by TUR111 but you're paying for 100%, increasing effective price)
NB.mechanism = 'consensus'
NB.description = 'Centralized_Dispatcher supplier node'
NB.maximumPower = 100000
NB.minimumPower = 0
NB.name = 'Avista'

Centralized_Dispatcher_Model = NeighborModel()
NBM = Centralized_Dispatcher_Model
NBM.name = 'Centralized_Dispatcher_Model'
NBM.converged = False
NBM.convergenceThreshold = 0.02
NBM.effectiveImpedance = 0.0
NBM.friend = False
NBM.transactive = False
# set default vertices using integration method, production_cost_from_vertices() helper function which does square law for losses
default_vertices = [Vertex(marginal_price=0.0551, prod_cost = 0, power=0, continuity=True, power_uncertainty=0.0), Vertex(marginal_price=0.05511, prod_cost = 551.1, power=100000, continuity=True, power_uncertainty=0.0)]
NBM.defaultVertices = [default_vertices]
NBM.activeVertices = [[]]
for t in ti:
    NBM.activeVertices[0].append(IntervalValue(NBM, t, Centralized_Dispatcher, MeasurementType.ActiveVertex, default_vertices[0]))
    NBM.activeVertices[0].append(IntervalValue(NBM, t, Centralized_Dispatcher, MeasurementType.ActiveVertex, default_vertices[1]))
# NBM.defaultVertices = [[]]
# for t in ti:
#     NBM.defaultVertices[0].append(IntervalValue(NBM, t, Avista, MeasurementType.ActiveVertex, default_vertices[0]))
#     NBM.defaultVertices[0].append(IntervalValue(NBM, t, Avista, MeasurementType.ActiveVertex, default_vertices[1]))
# NBM.activeVertices = NBM.defaultVertices
NBM.productionCosts = [[prod_cost_from_vertices(NBM, t, 0, energy_type=MeasurementType.PowerReal, market=dayAhead) for t in ti]]
NBM.object = NB
NB.model = NBM
Centralized_Dispatcher = NB
Centralized_Dispatcher_Model = NBM

#create list of transactive neighbors to my transactive node
mTN.neighbors = [Centralized_Dispatcher]

##################### Asset  Modelling  ########################################
## instantiate each LocalAsset and its LocalAssetModel
# a LocalAsset is "owned" by myTransactiveNode and is managed
# and represented by a LocalAssetModel. There must be a one to one
# correspondence between a model and its asset
##################### Asset  #1  ###############################################
SCUE = LocalAsset()
LA = SCUE
LA.name = 'SCUE'
LA.description = 'Smith Center for Undergraduate Education: a new building with flexible loads'
LA.maximumPower = [0,0,0] # load is negative power [kW] and this building has no production capacity
LA.minimimumPower = [-10000, -10000, -10000]

SCUEModel = FlexibleBuilding()
LAM = SCUEModel
LAM.name = 'SCUE'
LAM.defaultPower = [-100.0, -100.0, -100.0]
LAM.thermalAuction = [Centralized_Dispatcher]
#LAM.create_default_vertices(ti, dayAhead)# when creating flexibility, the first vertex should be largest demand
LAM.get_vertices_from_CESI_building(dayAhead)
LA.model = LAM
LAM.object = LA
SCUE = LA
SCUEModel = LAM
################################################################################
##################### Asset  #2  ###############################################
Johnson_Hall = LocalAsset()
LA = Johnson_Hall
LA.name = 'JohnsonHall'
LA.description = 'Inflexible buildings with electric, heating, and cooling loads'
LA.maximumPower = [0,0,0]
LA.minimumPower = [-1000,-1000,-1000]

Johnson_Hall_Model = InflexibleBuilding()
LAM = Johnson_Hall_Model
LAM.name = 'JohnsonHall'
LAM.defaultPower = [-100.0, -100.0, -100.0]
LAM.thermalAuction = [Centralized_Dispatcher]
LAM.get_vertices_from_inflexible_CESI_building(dayAhead)

LA.model = LAM
LAM.object = LA
Johnson_Hall = LA
Johnson_Hall_Model = LAM
################################################################################
##################### Asset  #3  ###############################################
Vault3_Building = LocalAsset()
LA = Vault3_Building
LA.name = 'Vault3'
LA.description = 'Inflexible buildings with electric, heating, and cooling loads'
LA.maximumPower = [0,0,0]
LA.minimumPower = [-1000,-1000,-1000]

Vault3_Building_Model = InflexibleBuilding()
LAM = Vault3_Building_Model
LAM.name = 'Vault3'
LAM.defaultPower = [-100.0, -100.0, -100.0]
LAM.thermalAuction = [Centralized_Dispatcher]
LAM.get_vertices_from_inflexible_CESI_building(dayAhead)

LA.model = LAM
LAM.object = LA
Vault3_Building = LA
Vault3_Building_Model = LAM
################################################################################
##################### Asset  #4  ###############################################
Vault5_Building = LocalAsset()
LA = Vault5_Building
LA.name = 'Vault5'
LA.description = 'Inflexible buildings with electric, heating, and cooling loads'
LA.maximumPower = [0,0,0]
LA.minimumPower = [-1000,-1000,-1000]

Vault5_Building_Model = InflexibleBuilding()
LAM = Vault5_Building_Model
LAM.name = 'Vault5'
LAM.defaultPower = [-100.0, -100.0, -100.0]
LAM.thermalAuction = [Centralized_Dispatcher]
LAM.get_vertices_from_inflexible_CESI_building(dayAhead)

LA.model = LAM
LAM.object = LA
Vault5_Building = LA
Vault5_Building_Model = LAM
################################################################################
##################### Asset  #5  ###############################################
TVW131_Buildings = LocalAsset()
LA = TVW131_Buildings
LA.name = 'TVW131'
LA.description = 'Inflexible buildings with electric, heating, and cooling loads'
LA.maximumPower = [0,0,0]
LA.minimumPower = [-1000,-1000,-1000]

TVW131_Building_Models = InflexibleBuilding()
LAM = TVW131_Building_Models
LAM.name = 'TVW131'
LAM.defaultPower = [-100.0, -100.0, -100.0]
LAM.thermalAuction = [Centralized_Dispatcher]
LAM.get_vertices_from_inflexible_CESI_building(dayAhead)

LA.model = LAM
LAM.object = LA
TVW131_Buildings = LA
TVW131_Building_Models = LAM
################################################################################
############################   Asset  #6  ######################################
TUR117_Buildings = LocalAsset()
LA = TUR117_Buildings
LA.name = 'TUR117'
LA.description = 'Inflexible buildings with electric, heating, and cooling loads'
LA.maximumPower = [0,0,0]
LA.minimumPower = [-1000,-1000,-1000]

TUR117_Building_Models = InflexibleBuilding()
LAM = TUR117_Building_Models
LAM.name = 'TUR117'
LAM.defaultPower = [-100.0, -100.0, -100.0]
LAM.thermalAuction = [Centralized_Dispatcher]
LAM.get_vertices_from_inflexible_CESI_building(dayAhead)

LA.model = LAM
LAM.object = LA
TUR117_Buildings = LA
TUR117_Building_Models = LAM
################################################################################
############################   Asset  #7  ######################################
TUR115_Buildings = LocalAsset()
LA = TUR115_Buildings
LA.name = 'TUR115'
LA.description = 'Inflexible buildings with electric, heating, and cooling loads'
LA.maximumPower = [0,0,0]
LA.minimumPower = [-1000,-1000,-1000]

TUR115_Building_Models = InflexibleBuilding()
LAM = TUR115_Building_Models
LAM.name = 'TUR115'
LAM.defaultPower = [-100.0, -100.0, -100.0]
LAM.thermalAuction = [Centralized_Dispatcher]
LAM.get_vertices_from_inflexible_CESI_building(dayAhead)

LA.model = LAM
LAM.object = LA
TUR115_Buildings = LA
TUR115_Building_Models = LAM
################################################################################
############################   Asset  #8  ######################################
TUR111_Buildings = LocalAsset()
LA = TUR111_Buildings
LA.name = 'TUR111'
LA.description = 'Inflexible buildings with electric, heating, and cooling loads'
LA.maximumPower = [0,0,0]
LA.minimumPower = [-1000,-1000,-1000]

TUR111_Building_Models = InflexibleBuilding()
LAM = TUR111_Building_Models
LAM.name = 'TUR111'
LAM.defaultPower = [-100.0, -100.0, -100.0]
LAM.thermalAuction = [Centralized_Dispatcher]
LAM.get_vertices_from_inflexible_CESI_building(dayAhead)

LA.model = LAM
LAM.object = LA
TUR111_Buildings = LA
TUR111_Building_Models = LAM
################################################################################
############################   Asset  #9  ######################################
SPU125_Buildings = LocalAsset()
LA = SPU125_Buildings
LA.name = 'SPU125'
LA.description = 'Inflexible buildings with electric, heating, and cooling loads'
LA.maximumPower = [0,0,0]
LA.minimumPower = [-1000,-1000,-1000]

SPU125_Building_Models = InflexibleBuilding()
LAM = SPU125_Building_Models
LAM.name = 'SPU125'
LAM.defaultPower = [-100.0, -100.0, -100.0]
LAM.thermalAuction = [Centralized_Dispatcher]
LAM.get_vertices_from_inflexible_CESI_building(dayAhead)

LA.model = LAM
LAM.object = LA
SPU125_Buildings = LA
SPU125_Building_Models = LAM
################################################################################
############################   Asset  #10  ######################################
EA7_JBA_Building = LocalAsset()
LA = EA7_JBA_Building
LA.name = 'EA7_JBA'
LA.description = 'Inflexible buildings with electric, heating, and cooling loads'
LA.maximumPower = [0,0,0]
LA.minimumPower = [-1000,-1000,-1000]

EA7_JBA_Building_Model = InflexibleBuilding()
LAM = EA7_JBA_Building_Model
LAM.name = 'EA7_JBA'
LAM.defaultPower = [-100.0, -100.0, -100.0]
LAM.thermalAuction = [Centralized_Dispatcher]
LAM.get_vertices_from_inflexible_CESI_building(dayAhead)

LA.model = LAM
LAM.object = LA
EA7_JBA_Building = LA
EA7_JBA_Building_Model = LAM
################################################################################
############################   Asset  #11  ######################################
EA7_SPA_Building = LocalAsset()
LA = EA7_SPA_Building
LA.name = 'EA7_SPA'
LA.description = 'Inflexible buildings with electric, heating, and cooling loads'
LA.maximumPower = [0,0,0]
LA.minimumPower = [-1000,-1000,-1000]

EA7_SPA_Building_Model = InflexibleBuilding()
LAM = EA7_SPA_Building_Model
LAM.name = 'EA7_SPA'
LAM.defaultPower = [-100.0, -100.0, -100.0]
LAM.thermalAuction = [Centralized_Dispatcher]
LAM.get_vertices_from_inflexible_CESI_building(dayAhead)

LA.model = LAM
LAM.object = LA
EA7_SPA_Building = LA
EA7_SPA_Building_Model = LAM
################################################################################
############################   Asset  #12  ######################################
EB13_JBA_Building = LocalAsset()
LA = EB13_JBA_Building
LA.name = 'EB13_JBA'
LA.description = 'Inflexible buildings with electric, heating, and cooling loads'
LA.maximumPower = [0,0,0]
LA.minimumPower = [-1000,-1000,-1000]

EB13_JBA_Building_Model = InflexibleBuilding()
LAM = EB13_JBA_Building_Model
LAM.name = 'EB13_JBA'
LAM.defaultPower = [-100.0, -100.0, -100.0]
LAM.thermalAuction = [Centralized_Dispatcher]
LAM.get_vertices_from_inflexible_CESI_building(dayAhead)

LA.model = LAM
LAM.object = LA
EB13_JBA_Building = LA
EB13_JBA_Building_Model = LAM
################################################################################
############################   Asset  #13  ######################################
EB13_JBB_Building = LocalAsset()
LA = EB13_JBB_Building
LA.name = 'EB13_JBB'
LA.description = 'Inflexible buildings with electric, heating, and cooling loads'
LA.maximumPower = [0,0,0]
LA.minimumPower = [-1000,-1000,-1000]

EB13_JBB_Building_Model = InflexibleBuilding()
LAM = EB13_JBB_Building_Model
LAM.name = 'EB13_JBB'
LAM.defaultPower = [-100.0, -100.0, -100.0]
LAM.thermalAuction = [Centralized_Dispatcher]
LAM.get_vertices_from_inflexible_CESI_building(dayAhead)
# LAM.productionCosts = [[prod_cost_from_vertices(LAM, t, 0, energy_type=MeasurementType.PowerReal, market=dayAhead) for t in ti],\
#     [prod_cost_from_vertices(LAM, t, 1, energy_type=MeasurementType.Heat, market=dayAhead) for t in ti],\
#         [prod_cost_from_vertices(LAM, t, 1, energy_type=MeasurementType.Cooling, market=dayAhead) for t in ti]]
LA.model = LAM
LAM.object = LA
EB13_JBB_Building = LA
EB13_JBB_Building_Model = LAM
################################################################################
##################### Asset  #14  ###############################################
# add diesel gen
gast1 = LocalAsset()
gast1.name = 'GasTurbine1'
gast1.description = 'Gas Turbine in CASP-A EAST'

gt1Model = GasTurbine()
gt1Model.name = 'GasTurbine1'
gt1Model.thermalAuction = [Centralized_Dispatcher]
gt1Model.size = 2500
gt1Model.ramp_rate = 1.3344e3
gt1Model.eff = 0.34423
gt1Model.datafilename = 'gt1'
gt1Model.make_fit_curve()
gt1Model.get_vertices_from_linear_model(dayAhead)
gast1.model = gt1Model
gt1Model.object = gast1
################################################################################
##################### Asset  #15 ###############################################
# add diesel gen
gast2 = LocalAsset()
gast2.name = 'GasTurbine2'
gast2.description = 'Gas Turbine at GWSP'

gt2Model = GasTurbine()
gt2Model.name = 'GasTurbine2'
gt2Model.thermalAuction = [Centralized_Dispatcher]
gt2Model.size = 2187.5
gt2Model.ramp_rate = 1.3344e3
gt2Model.eff = 0.34827
gt2Model.datafilename = 'gt2'
gt2Model.make_fit_curve()
gt2Model.get_vertices_from_linear_model(dayAhead)
gast2.model = gt2Model
gt1Model.object = gast2
################################################################################
##################### Asset  #16 ###############################################
# add diesel gen
gast3 = LocalAsset()
gast3.name = 'GasTurbine3'
gast3.description = 'Gas Turbine at GWSP'

gt3Model = GasTurbine()
gt3Model.name = 'GasTurbine3'
gt3Model.thermalAuction = [Centralized_Dispatcher]
gt3Model.size = 1375
gt3Model.ramp_rate = 1.3344e3
gt3Model.eff = 0.34517
gt3Model.datafilename = 'gt2'
gt3Model.make_fit_curve()
gt3Model.get_vertices_from_linear_model(dayAhead)
gast3.model = gt3Model
gt3Model.object = gast3
################################################################################
##################### Asset  #17 ###############################################
# add Gas Turbine
gast4 = LocalAsset()
gast4.name = 'GasTurbine4'
gast4.description = 'Gas Turbine at GWSP'

gt4Model = GasTurbine()
gt4Model.name = 'GasTurbine4'
gt4Model.thermalAuction = [Centralized_Dispatcher]
gt4Model.size = 1375
gt4Model.ramp_rate = 1.3344e3
gt4Model.eff = 0.34817
gt4Model.datafilename = 'gt2'
gt4Model.make_fit_curve()
gt4Model.get_vertices_from_linear_model(dayAhead)
gast4.model = gt4Model
gt4Model.object = gast4
################################################################################
##################### Asset  #18  ###############################################
# Add boilers 1
boiler1 = LocalAsset()
boiler1.name = 'Boiler1'
boiler1.description = '1st boiler at CASP'

boiler1Model = Boiler()
boiler1Model.name = 'Boiler1'
boiler1Model.size = 20000
boiler1Model.ramp_rate = 1333.3
boiler1Model.eff = 0.99
boiler1Model.thermalAuction = [Centralized_Dispatcher]
boiler1Model.make_fit_curve()
boiler1Model.get_vertices_from_linear_model(dayAhead)
boiler1.model = boiler1Model
boiler1Model.object = boiler1
################################################################################
########################## Asset  #19, #20,#21,#22, ##########################
# add boiler 2
boiler2 = copy.deepcopy(boiler1)
boiler2.name = 'Boiler2'
boiler2.model.name = 'Boiler2'
boiler2.description = '2nd boiler at CASP'

boiler3 = copy.deepcopy(boiler1)
boiler3.name = 'Boiler3'
boiler3.model.name = 'Boiler3'
boiler3.description = '3rd boiler at CASP'

boiler4 = copy.deepcopy(boiler1)
boiler4.name = 'Boiler4'
boiler4.model.name = 'Boiler4'
boiler4.description = '4th boiler at CASP'

boiler5 = copy.deepcopy(boiler1)
boiler5.name = 'Boiler5'
boiler5.model.name = 'Boiler5'
boiler5.description = '5th boiler at CASP'
################################################################################
##################### Asset  #23  ##############################################
# add Gas Turbine
chiller1 = LocalAsset()
chiller1.name = 'Chiller1'
chiller1.description = 'Carrier Chiller1'

chiller1Model = Chiller()
chiller1Model.name = 'Chiller1'
chiller1Model.thermalAuction = [Centralized_Dispatcher]
chiller1Model.size = 7.279884675000000e+03
chiller1Model.ramp_rate = 4.8533e3
chiller1Model.eff = 5.874
chiller1Model.datafilename = 'carrierchiller1'
chiller1Model.make_fit_curve()
chiller1Model.get_vertices_from_linear_model(dayAhead)
chiller1.model = chiller1Model
chiller1Model.object = chiller1
################################################################################
##################### Asset  #24  ##############################################
chiller2 = LocalAsset()
chiller2.name = 'Chiller2'
chiller2.description = 'york chiller 1'

chiller2Model = Chiller()
chiller2Model.name = 'Chiller2'
chiller2Model.thermalAuction = [Centralized_Dispatcher]
chiller2Model.size = 5.268245045000001e+03
chiller2Model.ramp_rate = 3.5122e3
chiller2Model.eff = 4.689
chiller2Model.datafilename = 'yorkchiller1'
chiller2Model.make_fit_curve()
chiller2Model.get_vertices_from_linear_model(dayAhead)
chiller2.model = chiller2Model
chiller2Model.object = chiller2
################################################################################
##################### Asset  #25  ##############################################
chiller3 = LocalAsset()
chiller3.name = 'Chiller3'
chiller3.description = 'york chiller 3'

chiller3Model = Chiller()
chiller3Model.name = 'Chiller3'
chiller3Model.thermalAuction = [Centralized_Dispatcher]
chiller3Model.size = 5.268245045000001e+03
chiller3Model.ramp_rate = 3.5122e3
chiller3Model.eff = 4.689
chiller3Model.datafilename = 'yorkchiller3'
chiller3Model.make_fit_curve()
chiller3Model.get_vertices_from_linear_model(dayAhead)
chiller3.model = chiller3Model
chiller3Model.object = chiller3
################################################################################
##################### Asset  #26  ##############################################
chiller4 = LocalAsset()
chiller4.name = 'Chiller4'
chiller4.description = 'carrier chiller 2'

chiller4Model = Chiller()
chiller4Model.name = 'Chiller4'
chiller4Model.thermalAuction = [Centralized_Dispatcher]
chiller4Model.size = 4.853256450000000e+03
chiller4Model.ramp_rate = 3.2355e3
chiller4Model.eff = 9.769
chiller4Model.datafilename = 'carrierchiller2'
chiller4Model.make_fit_curve()
chiller4Model.get_vertices_from_linear_model(dayAhead)
chiller4.model = chiller4Model
chiller4Model.object = chiller4
################################################################################
##################### Asset  #27  ##############################################
chiller5 = LocalAsset()
chiller5.name = 'Chiller5'
chiller5.description = 'Carrier Chiller3'

chiller5Model = Chiller()
chiller5Model.name = 'Chiller5'
chiller5Model.thermalAuction = [Centralized_Dispatcher]
chiller5Model.size = 4.853256450000000e+03
chiller5Model.ramp_rate = 3.2355e3
chiller5Model.eff = 9.769
chiller5Model.datafilename = 'carrierchiller3'
chiller5Model.make_fit_curve()
chiller5Model.get_vertices_from_linear_model(dayAhead)
chiller5.model = chiller5Model
chiller5Model.object = chiller5
################################################################################
##################### Asset  #28  ##############################################
chiller6 = LocalAsset()
chiller6.name = 'Chiller6'
chiller6.description = 'Carrier Chiller4'

chiller6Model = Chiller()
chiller6Model.name = 'Chiller6'
chiller6Model.thermalAuction = [Centralized_Dispatcher]
chiller6Model.size = 1.758426250000000e+03
chiller6Model.ramp_rate = 1.1723e3
chiller6Model.eff = 1.5337
chiller6Model.datafilename = 'carrierchiller4'
chiller6Model.make_fit_curve()
chiller6Model.get_vertices_from_linear_model(dayAhead)
chiller6.model = chiller6Model
chiller6Model.object = chiller6
################################################################################
##################### Asset  #29  ##############################################
chiller7 = LocalAsset()
chiller7.name = 'Chiller7'
chiller7.description = 'Carrier Chiller7'

chiller7Model = Chiller()
chiller7Model.name = 'Chiller7'
chiller7Model.thermalAuction = [Centralized_Dispatcher]
chiller7Model.size = 5.275278750000000e+03
chiller7Model.ramp_rate = 3.5169e3
chiller7Model.eff = 5.506
chiller7Model.datafilename = 'carrierchiller7'
chiller7Model.make_fit_curve()
chiller7Model.get_vertices_from_linear_model(dayAhead)
chiller7.model = chiller7Model
chiller7Model.object = chiller7
################################################################################
##################### Asset  #30  ##############################################
chiller8 = LocalAsset()
chiller8.name = 'Chiller8'
chiller8.description = 'Carrier Chiller8'

chiller8Model = Chiller()
chiller8Model.name = 'Chiller8'
chiller8Model.thermalAuction = [Centralized_Dispatcher]
chiller8Model.size = 5.275278750000000e+03
chiller8Model.ramp_rate = 3.5169e3
chiller8Model.eff = 5.506
chiller8Model.datafilename = 'carrierchiller8'
chiller8Model.make_fit_curve()
chiller8Model.get_vertices_from_linear_model(dayAhead)
chiller8.model = chiller8Model
chiller8Model.object = chiller8
################################################################################
##################### Asset  #31  ##############################################
chiller9 = LocalAsset()
chiller9.name = 'Chiller9'
chiller9.description = 'Trane Chiller'

chiller9Model = Chiller()
chiller9Model.name = 'Chiller9'
chiller9Model.thermalAuction = [Centralized_Dispatcher]
chiller9Model.size = 1.415462794200000e+03
chiller9Model.ramp_rate = 943.6419
chiller9Model.eff = 4.56734
chiller9Model.datafilename = 'tranechiller'
chiller9Model.make_fit_curve()
chiller9Model.get_vertices_from_linear_model(dayAhead)
chiller9.model = chiller9Model
chiller9Model.object = chiller9


################################################################################
mTN.localAssets = [SCUE, Johnson_Hall, Vault3_Building, Vault5_Building, TVW131_Buildings, TUR117_Buildings, TUR115_Buildings, TUR111_Buildings, SPU125_Buildings, EA7_JBA_Building, EA7_SPA_Building, EB13_JBA_Building, EB13_JBB_Building, \
                   gast1, gast2, gast3, gast4, boiler1, boiler2, boiler3, boiler4, boiler5, chiller1, chiller2, chiller3, chiller4, chiller5, chiller6, chiller7, chiller8, chiller9]
################################################################################

## Additional setup script
# the following methods would normally be called soon after the above script
# to launch the system
# call the Market method that will instantiate active future time intervals

#############################################################################
########################## Intialize Helics  ################################
broker = create_broker()
name_neighbors = []
#json_filename = create_config_for_helics(mTN.name, [mTN.neighbors[i].name for i in range(len(mTN.neighbors))])
#json_filename = create_config_for_helics(dayAhead.name, [mTN.localAssets[0].name], [mTN.localAssets[i].name for i in range(len(mTN.localAssets))], [3 for i in range(len(mTN.localAssets))], config_for_gridlabd = True)
json_filename = create_config_for_helics(dayAhead.name, [mTN.localAssets[0].name], [mTN.localAssets[i].name for i in range(len(mTN.localAssets))], [3 for i in range(len(mTN.localAssets))], config_for_gridlabd = True)

print(json_filename)
fed = register_federate(json_filename)
status = h.helicsFederateEnterInitializingMode(fed)
status = h.helicsFederateEnterExecutingMode(fed)

##############################################################################

dayAhead.intervalsToClear = 24
for time in range(dayAhead.intervalsToClear):

    for asset in mTN.localAssets:
        if 'FlexibleBuilding' in str(asset.model):
            asset.model.request_helics_to_get_vertices_from_CESI_building(dayAhead, fed)
            #asset.model.get_vertices_from_CESI_building(dayAhead)
        if 'InflexibleBuilding' in str(asset.model):
            asset.model.get_vertices_from_inflexible_CESI_building(dayAhead)

    dayAhead.check_intervals()
    dayAhead.centralized_dispatch(WSU_Campus)

    #dayAhead.update_electrical_network(mTN, fed)
    for asset in mTN.localAssets:
        asset.model.update_dispatch(dayAhead, fed)

    dayAhead.marketClearingTime = dayAhead.marketClearingTime + timedelta(hours=1)
    dayAhead.nextMarketClearingTime = dayAhead.marketClearingTime + timedelta(hours=1)

# call the information service that predicts and stores outdoor temps
PullmanTemperatureForecast.update_information(dayAhead)

############################ Finalize Helics  ##############################
destroy_federate(fed)
############################################################################


