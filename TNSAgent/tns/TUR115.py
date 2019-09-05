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
from market import Market
from market_state import MarketState
from auction import Auction
from inflexible_building import InflexibleBuilding
from gas_turbine import GasTurbine
from boiler import Boiler
from helpers import prod_cost_from_vertices
from vertex import Vertex
from interval_value import IntervalValue

# create a neighbor model 
TUR115 = myTransactiveNode()
mTN = TUR115
mTN.description = 'substation TUR115 feeds half of campus ave steam plant\
                    including one CHP generator and some of the west campus buildings'
mTN.name = 'T115'

# set up AVISTA power meter
TUR115_meter = MeterPoint()
MP = TUR115_meter
MP.description = 'meters SCUE building electric use from AVISTA'
MP.measurementType = MeasurementType.PowerReal
MP.measurement = MeasurementUnit.kWh
TUR115_meter = MP

# provide a cell array of all the MeterPoints to myTransactiveNode
mTN.meterpoints = [TUR115_meter]

# instantiate each information service model
# this is services that can be queried for information
# this includes model prediction for future time intervals
# Pullman Temperature Forecast <-- Information service model
PullmanTemperatureForecast = TemperatureForecastModel()
ISM = PullmanTemperatureForecast
ISM.name = 'PullmanTemperatureForecast'
ISM.predictedValues = [] # dynamically assigned

mTN.informationServiceModels = [PullmanTemperatureForecast]

##################################################################
## Instantiate Markets and Auctions
# Markets specify active TimeIntervals
# Auctions handle thermal loads

## Day Ahead Market
dayAhead = Market(measurementType = [MeasurementType.PowerReal, MeasurementType.Heat, MeasurementType.Cooling])
MKT = dayAhead
MKT.name = 'T115_Market'
MKT.commitment = False # start without having commited any resources
MKT.converged = False # start without having converged
MKT.defaultPrice = [0.04, 0.02, 0.03] # [$/kWh]
MKT.dualityGapThreshold = 0.001 #optimal convergence within 0.1Wh
MKT.futureHorizon = timedelta(hours=24)
MKT.intervalDuration = timedelta(hours=1)
MKT.intervalsToClear = 24
MKT.marketClearingTime = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) # align with top of hour
MKT.marketOrder = 1 # this is the first and only market
MKT.nextMarketClearingTime = MKT.marketClearingTime + timedelta(hours=1)
MKT.initialMarketState = MarketState.Inactive
dayAhead = MKT
dayAhead.check_intervals()

ti = dayAhead.timeIntervals

# Thermal Auctions are seen as neighbor nodes

mTN.markets = [dayAhead]

####################################################################################

## Instantiate Neighbors and NeighborModels
Avista = Neighbor()
NB = Avista
NB.lossFactor = 0.01 # one percent loss at full power (only 99% is seen by TUR111 but you're paying for 100%, increasing effective price)
NB.mechanism = 'consensus'
NB.description = 'Avista electricity supplier node'
NB.maximumPower = 100000
NB.minimumPower = 0
NB.name = 'Avista'

AvistaModel = NeighborModel()
NBM = AvistaModel
NBM.name = 'Avista_model'
NBM.converged = False
NBM.convergenceThreshold = 0.02
NBM.effectiveImpedance = 0.0
NBM.friend = False
NBM.transactive = True
# set default vertices using integration method, production_cost_from_vertices() helper function which does square law for losses
default_vertices = [Vertex(marginal_price=0.029, prod_cost = 0, power=0, continuity=True, power_uncertainty=0.0), Vertex(marginal_price=0.031, prod_cost = 300.0, power=100000, continuity=True, power_uncertainty=0.0)]
NBM.defaultVertices = [default_vertices]
NBM.activeVertices = [[]]
for t in ti:
    NBM.activeVertices[0].append(IntervalValue(NBM, t, Avista, MeasurementType.ActiveVertex, default_vertices[0]))
    NBM.activeVertices[0].append(IntervalValue(NBM, t, Avista, MeasurementType.ActiveVertex, default_vertices[1]))
# NBM.defaultVertices = [[]]
# for t in ti:
#     NBM.defaultVertices[0].append(IntervalValue(NBM, t, Avista, MeasurementType.ActiveVertex, default_vertices[0]))
#     NBM.defaultVertices[0].append(IntervalValue(NBM, t, Avista, MeasurementType.ActiveVertex, default_vertices[1]))
# NBM.activeVertices = NBM.defaultVertices
NBM.productionCosts = [[prod_cost_from_vertices(NBM, t, 0, energy_type=MeasurementType.PowerReal, market=dayAhead) for t in ti]]
NBM.object = NB
NB.model = NBM
Avista = NB
AvistaModel = NBM 

# define thermal auctions here
# thermal auctions are neighbors which only interact with thermal energy
# steam auction
SteamLoop = Neighbor()
NB = SteamLoop
NB.lossFactor = 0.01
NB.mechanism = 'consensus'
NB.description = 'district heating steam distribution loop'
NB.maximumPower = 100000
NB.minimumPower = -10000
NB.name = 'steam_loop'
NB.Tsupply = 250
NB.Treturn = 120
NB.naturalGasPrice = 0.01

HeatAuctionModel = NeighborModel(measurementType=[MeasurementType.Heat])
NBM = HeatAuctionModel
NBM.name = 'steam_loop_model'
NBM.converged = False
NBM.convergenceThreshold = 0.02
NBM.effectiveImpedance = [0.0]
NBM.friend = True
NBM.transactive = True
default_vertices =[Vertex(marginal_price=-0.01, prod_cost = 0, power=-10000, continuity=True, power_uncertainty=0.0), Vertex(marginal_price=0.01, prod_cost = 100.0, power=10000, continuity=True, power_uncertainty=0.0)]
NBM.defaultVertices =  [default_vertices]#[[IntervalValue(NBM, t, HeatAuctionModel, MeasurementType.ActiveVertex, vert) for t in ti] for vert in default_vertices]
NBM.activeVertices =  [[IntervalValue(NBM, t, HeatAuctionModel, MeasurementType.ActiveVertex, vert) for t in ti] for vert in default_vertices]
NBM.productionCosts = [[prod_cost_from_vertices(NBM, t, 0, energy_type=MeasurementType.Heat, market=dayAhead) for t in ti]]

NBM.object = NB
NB.model = NBM
SteamLoop = NB
HeatAuctionModel = NBM

# cold water auction
ColdWaterLoop = Neighbor()
NB = ColdWaterLoop
NB.lossFactor = 0.01
NB.mechanism = 'consensus'
NB.description = 'district cooling cold water loop'
NB.maximumPower = 100000
NB.minimumPower = -10000
NB.name = 'water_loop'
NB.Tsupply = 4
NB.Treturn = 15

CoolAuctionModel = NeighborModel(measurementType=[MeasurementType.Cooling])
NBM = CoolAuctionModel
NBM.name = 'water_loop_model'
NBM.converged = False
NBM.convergenceThreshold = 0.02
NBM.effectiveImpedance = [0.0]
NBM.friend = True
NBM.transactive = True
default_vertices = [Vertex(marginal_price=-0.02, prod_cost = 0, power=-10000, continuity=True, power_uncertainty=0.0), Vertex(marginal_price=0.02, prod_cost = 200.0, power=10000, continuity=True, power_uncertainty=0.0)]
NBM.defaultVertices =  [default_vertices]#[[IntervalValue(NBM, t, CoolAuctionModel, MeasurementType.ActiveVertex, vert) for t in ti] for vert in default_vertices]#, Vertex(marginal_price=0.02, prod_cost = 300.0, power=10000, continuity=True, power_uncertainty=0.0)]]
NBM.activeVertices = [[IntervalValue(NBM, t, CoolAuctionModel, MeasurementType.ActiveVertex, vert) for t in ti] for vert in default_vertices]
NBM.productionCosts = [[prod_cost_from_vertices(NBM, t, 0, energy_type=MeasurementType.Cooling, market=dayAhead) for t in ti]]

NBM.object = NB
NB.model = NBM
ColdWaterLoop = NB
CoolAuctionModel = NBM

#create list of transactive neighbors to my transactive node
mTN.neighbors = [Avista, SteamLoop, ColdWaterLoop]


###########################################################################################
# instantiate each Local Asset and its LocalAssetModel
# a LocalAsset is "owned" by myTransactiveNode and is managed and 
# represented by a LocalAssetModel. There must be a one to one
# correspondence between a model and its asset

#add inflexible west campus buildings
WestCampusBuildings = LocalAsset()
LA = WestCampusBuildings
LA.name = 'WestCampus'
LA.description = 'Inflexible buildings with electric, heating, and cooling loads'
LA.maximumPower = [0,0,0]
LA.minimumPower = [-1000,-1000,-1000]

WCBModel = InflexibleBuilding()
LAM = WCBModel
LAM.name = 'WestCampus'
LAM.defaultPower = [-100.0, -100.0, 0.0]
LAM.thermalAuction = [SteamLoop, ColdWaterLoop]
LAM.update_active_vertex(ti, dayAhead)

LA.model = LAM
LAM.object = LA
WestCampusBuildings = LA
WCBModel = LAM

# add diesel gen
gen1 = LocalAsset()
gen1.name = 'gen1'
gen1.description = 'diesel generator at college ave steam plant'
gen1.maximumPower = [1000, 600]
gen1.minimumPower = [0,0]

gen1Model = GasTurbine()
gen1Model.name = 'gen1Model'
gen1Model.thermalAuction = [SteamLoop]
gen1Model.size = 1000
gen1Model.create_default_vertices(ti, dayAhead)
gen1Model.ramp_rate = 666.6692
gen1Model.productionCosts = [[prod_cost_from_vertices(gen1Model, t, 1, energy_type=MeasurementType.PowerReal, market =dayAhead) for t in ti], [prod_cost_from_vertices(gen1Model, t, 0.6, energy_type=MeasurementType.Heat, market=dayAhead) for t in ti]]
gen1.model = gen1Model
gen1Model.object = gen1

# add boilers 4 and 5
boiler4 = LocalAsset()
boiler4.name = 'Boiler4'
boiler4.description = '1st boiler at CASP'
boiler4.maximumPower = [20000]
boiler4.minimumPower = [0]

boiler4Model = Boiler()
boiler4Model.name = 'Boiler4'
boiler4Model.size = 20000
boiler4Model.ramp_rate = 1333.3
boiler4Model.thermalAuction = [SteamLoop]
boiler4Model.create_default_vertices(ti, dayAhead)
boiler4Model.productionCosts = [[prod_cost_from_vertices(boiler4Model, t, 1, energy_type=MeasurementType.Heat, market=dayAhead) for t in ti]]
boiler4.model = boiler4Model
boiler4Model.object = boiler4

# add boiler 3
boiler5 = LocalAsset()
boiler5.name = 'Boiler5'
boiler5.description = '2nd boiler at CASP'
boiler5.maximumPower = [20000]
boiler5.minimumPower = [0]

boiler5Model = Boiler()
boiler5Model.name = 'Boiler5'
boiler5Model.size = 20000
boiler5Model.ramp_rate = 1333.3
boiler5Model.thermalAuction = [SteamLoop]
boiler5Model.create_default_vertices(ti, dayAhead)
boiler5Model.productionCosts = [[prod_cost_from_vertices(boiler5Model, t, 1, energy_type=MeasurementType.Heat, market=dayAhead) for t in ti]]
boiler5.model = boiler5Model
boiler5Model.object = boiler5


mTN.localAssets = [WestCampusBuildings, gen1, boiler4, boiler5]


#############################################################################
## Additional setup script
# the following methods would normally be called soon after the above script
# to launch the system
# 
# call the Market method that will instantiate active future time intervals
dayAhead.check_intervals()

# call the information service that predicts and stores outdoor temps
PullmanTemperatureForecast.update_information(dayAhead)

# recieve any transactive signals sent to myTransactiveNode from its
# TransactiveNeighbors.
AvistaModel.receive_transactive_signal(TUR115)
HeatAuctionModel.receive_transactive_signal(TUR115)
CoolAuctionModel.receive_transactive_signal(TUR115)

#balance supply and demand at myTransactiveNode. This is iterative. A
# succession of iterationcounters and duality gap (the convergence metric)
# will be generated until the system converges. All scheduled powers and
# marginal prices should be meaningful for all active time intervals at the
# conclusion of this method
dayAhead.balance(TUR115)

# myTransactiveNode must prepare a set of TransactiveRecords for each of 
# its TransactiveNeighbors. The records are updated and stored into the
# property "mySignal" of the TransactiveNeighbor.
AvistaModel.prep_transactive_signal(dayAhead, TUR115)
HeatAuctionModel.prep_transactive_signal(dayAhead, TUR115)
CoolAuctionModel.prep_transactive_signal(dayAhead, TUR115)

# Finally, the prepared TransactiveRecords are sent to their corresponding
# TransactiveNeighbor.
AvistaModel.send_transactive_signal(TUR115)
HeatAuctionModel.send_transactive_signal(TUR115)
CoolAuctionModel.send_transactive_signal(TUR115)

# invoke the market object to sum all powers as will be needed by the 
# net supply/demand curve
dayAhead.assign_system_vertices(TUR115)

# view the system supply/demand curve
dayAhead.view_net_curve(0)
dayAhead.view_net_curve(0, energy_type=MeasurementType.Heat)
dayAhead.view_net_curve(0, energy_type=MeasurementType.Cooling)