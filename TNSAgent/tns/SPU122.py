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

# create a neighbor model
SPU122 = myTransactiveNode()
mTN = SPU122
mTN.description = 'substation SPU122 feeds half of the grimes way steam plant\
    as well as the central campus inflexible buildings'
mTN.name = 'S122'

# set up Avista power meter
SPU122_meter = MeterPoint()
MP = SPU122_meter
MP.description = 'meters SCUE building electric use from Avista'
MP.measurementType = MeasurementType.PowerReal
MP.measurement = MeasurementUnit.kWh
SPU122_meter = MP

# provide a cell array of all the MeterPoints to myTransactiveNode
mTN.meterPoints = [SPU122_meter]

# instantiate each information service model
# this is services that can be queried for information
# this includes model prediction for future time intervals
# Pullman Temperature Forecast <-- Information service model
PullmanTemperatureForecast = TemperatureForecastModel()
ISM = PullmanTemperatureForecast
ISM.name = 'PullmanTemperatureForecast'
ISM.predictedValues = [] # dynamically assigned

mTN.informationServiceModels = [PullmanTemperatureForecast]

########################################################################
## Instantiate Markets

## Day Ahead Market
dayAhead = Market(measurementType= [MeasurementType.PowerReal, MeasurementType.Heat, MeasurementType.Cooling])
MKT = dayAhead
MKT.name = 'S122_Market'
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

ti = dayAhead.timeIntervals[0]

# Thermal Auctions are seen as neighbor nodes

mTN.markets = [dayAhead]

############################################################################################################################
## Instantiate Neighbors and NeighborModels
Avista = Neighbor()
NB = Avista
NB.lossFactor = 0.01
NB.mechanism = 'consensus'
NB.description = 'Avista electricity supplier node'
NB.maximumPower = 100000
NB.minimumPower = 0
NB.name = 'Avista'

AvistaModel = NeighborModel()
NBM = AvistaModel
NBM.converged = False
NBM.convergenceThreshold = 0.02
NBM.effectiveImpedance = 0.0
NBM.friend = False
NBM.transactive = True
NBM.object = NB
NB.model = NBM
Avista = NB
AvistaModel = NBM 

# define thermal loops here
# thermal loops are neighbors which only interact with thermal energy
# steam loop
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
NBM.converged = False
NBM.convergenceThreshold = 0.02
NBM.effectiveImpedance = [0.0]
NBM.friend = True
NBM.transactive = True

NBM.object = NB
NB.model = NBM
SteamLoop = NB
HeatAuctionModel = NBM

# cold water loop
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
NBM.converged = False
NBM.convergenceThreshold = 0.02
NBM.effectiveImpedance = 0.0
NBM.friend = True
NBM.transactive = True

NBM.object = NB
NB.model = NBM
ColdWaterLoop = NB
CoolAuctionModel = NBM

# neighbor nodes are SPU124, and possibly SPU125
SPU124 = Neighbor()
NB = SPU124
NB.lossFactor = 0.01
NB.mechanism = 'consensus'
NB.description = 'substation that feeds the other half of GWSP, Johnson Hall, and the east campus inflexible buildings'
NB.maximumPower = 10000
NB.minimumPower = -10000
NB.name = 'S124'

SPU124Model = NeighborModel(measurementType=[MeasurementType.PowerReal, MeasurementType.Heat, MeasurementType.Cooling])
NBM = SPU124Model
NBM.converged = False
NBM.convergenceThreshold = 0.01
NBM.effectiveImpedance = 0.0
NBM.friend = True
NBM.transactive = True

NBM.object = NB
NB.model = NBM
SPU124 = NB
SPU124Model = NBM

# create list of transactive neighbors to my Transactive Node
mTN. neighbors = [Avista, SteamLoop, ColdWaterLoop, SPU124]

###########################################################################################################################
## Instantiate Local Assets and their LocalAssetModels
# a LocalAsset is "owned" by myTransactiveNode and is managed and 
# represented by a LocalAssetModel. There must be a one to one 
# correspondence between a model and its asset

# add inflexible central campus buildings
CentralCampusBuildings = LocalAsset()
LA = CentralCampusBuildings
LA.name = 'CentralCampus'
LA.description = 'Inflexible buildings with electric, heat, and cooling loads'
LA.maximumPower = [0, 0, 0]
LA.minimumPower = [-100000, -1000, -1000]

CCBModel = InflexibleBuilding()
CCBModel.name = 'CentralCampus'
CCBModel.defaultPower = [-1000.0, 0, 0]
CCBModel.thermalAuction = [SteamLoop, ColdWaterLoop]
CCBModel.update_active_vertex(ti, dayAhead)

LA.model = CCBModel
CentralCampusBuildings = LA
CCBModel.object = CentralCampusBuildings

# add gas turbine 1
gt1 = LocalAsset()
gt1.name = 'GasTurbine1'
gt1.description = 'First gas turbine at the grimes way steam plant. CHP GT'
gt1.maximumPower = [1000, 600]
gt1.minimumPower = [0, 0]

gt1Model = GasTurbine()
gt1Model.name = 'GasTurbine1'
#gt1Model.thermalAuction = heat_auction
gt1Model.size = 1000
gt1Model.ramp_rate = 1.3344e3
gt1Model.create_default_vertices()

gt1.model = gt1Model
gt1Model.object = gt1

# add boiler1
boiler1 = LocalAsset()
boiler1.name = 'Boiler1'
boiler1.description = 'boiler at grimes way steam plant on the side of SPU122'
boiler1.maximumPower = [20000]
boiler1.minimumPower = [0]

boiler1Model = Boiler(name ='boiler1', size =20000)
boiler1Model.ramp_rate = 1333.3
boiler1Model.create_default_vertices()

boiler1.model = boiler1Model
boiler1Model.object = boiler1

# make a list of local assets at SPU122
mTN.localAssets = [CentralCampusBuildings, gt1, boiler1]

##########################################################################################################
## Additional setup script here
# the following methods would normally be called soon after the above script
# to launch the system
# 
# call the Market method that will instantiate active future time intervals
dayAhead.check_intervals()

# call the information service that predicts and stores outdoor temps
PullmanTemperatureForecast.update_information(dayAhead)

# recieve any transactive signals sent to myTransactiveNode from its
# TransactiveNeighbors.
AvistaModel.receive_transactive_signal(SPU122)
HeatAuctionModel.receive_transactive_signal(SPU122)
CoolAuctionModel.receive_transactive_signal(SPU122)

#balance supply and demand at myTransactiveNode. This is iterative. A
# succession of iterationcounters and duality gap (the convergence metric)
# will be generated until the system converges. All scheduled powers and
# marginal prices should be meaningful for all active time intervals at the
# conclusion of this method
dayAhead.balance(SPU122)

# myTransactiveNode must prepare a set of TransactiveRecords for each of 
# its TransactiveNeighbors. The records are updated and stored into the
# property "mySignal" of the TransactiveNeighbor.
AvistaModel.prep_transactive_signal(dayAhead, SPU122)
HeatAuctionModel.prep_transactive_signal(dayAhead, SPU122)
CoolAuctionModel.prep_transactive_signal(dayAhead, SPU122)

# invoke the market object to sum all powers as will be needed by the 
# net supply/demand curve
dayAhead.assign_system_vertices(SPU122)

# view the system supply/demand curve
dayAhead.view_net_curve(0)
dayAhead.view_net_curve(0, energy_type=MeasurementType.Heat)
dayAhead.view_net_curve(0, energy_type=MeasurementType.Cooling)