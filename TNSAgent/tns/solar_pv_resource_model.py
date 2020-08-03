# -*- coding: utf-8 -*- {{{
# vim: set fenc=utf-8 ft=python sw=4 ts=4 sts=4 et:

# Copyright (c) 2017, Battelle Memorial Institute
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# 'AS IS' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation
# are those of the authors and should not be interpreted as representing
# official policies, either expressed or implied, of the FreeBSD
# Project.
#
# This material was prepared as an account of work sponsored by an
# agency of the United States Government.  Neither the United States
# Government nor the United States Department of Energy, nor Battelle,
# nor any of their employees, nor any jurisdiction or organization that
# has cooperated in the development of these materials, makes any
# warranty, express or implied, or assumes any legal liability or
# responsibility for the accuracy, completeness, or usefulness or any
# information, apparatus, product, software, or process disclosed, or
# represents that its use would not infringe privately owned rights.
#
# Reference herein to any specific commercial product, process, or
# service by trade name, trademark, manufacturer, or otherwise does not
# necessarily constitute or imply its endorsement, recommendation, or
# favoring by the United States Government or any agency thereof, or
# Battelle Memorial Institute. The views and opinions of authors
# expressed herein do not necessarily state or reflect those of the
# United States Government or any agency thereof.
#
# PACIFIC NORTHWEST NATIONAL LABORATORY
# operated by BATTELLE for the UNITED STATES DEPARTMENT OF ENERGY
# under Contract DE-AC05-76RL01830

# }}}


from helpers import *
from measurement_type import MeasurementType
from local_asset import LocalAsset
from local_asset_model import LocalAssetModel
from interval_value import IntervalValue
from measurement_type import MeasurementType
import glmanip
import os
import csv
import helics as h

class SolarPvResourceModel(LocalAssetModel, object):
    # SolarPvResourceModel Subclass for renewable solar PV generation The Solar
    # PV resource is treated here as a must-take resource. This is unlike
    # dispatchable resources in this regard. Production may be predicted.
    # The main features of this model are (1) the introduction of property
    # cloudFactor, an IntervalValue, that allows us to reduce the expected
    # solar generation according to cloud cover, and (2) method
    # solar_generation() that creates the envelope, best-case, power production
    # for the resource as a function of time-of-day.

    def __init__(self, name = None, size=0.0, energy_types = [MeasurementType.PowerReal]):
        super(SolarPvResourceModel, self).__init__()
        self.model_type = 'SolarPV'
        self.cloudFactor = 1.0
        self.measurementType = [MeasurementType.PowerReal]
        self.size = size
        self.thermalAuction = None
        self.energy_type = ['electricity']
        self.vertices = [[] for et in energy_types] # list of vertex class instances defining the efficiency curve
        self.activeVertices = [[] for et in energy_types] #list of Vertex class instances defining the cost vs. power used
        self.record = {}  # added by Nathan Gray to record simulation results.
        # self.record is filled in the update_dispatch method.

    def make_solar_forecast(self, mkt):

        path = os.getcwd()
        os.chdir(path + '/Solar_data')
        path_solar = os.getcwd()
        dir_for_glm = (path_solar + '/solar.glm')
        print("Configuring GridLAB-D according to Simulation date")
        glm_lines = glmanip.read(dir_for_glm, path_solar, buf=[])
        [model, clock, directives, modules, classes] = glmanip.parse(glm_lines)
        clock['starttime'] = "'" + str(mkt.marketClearingTime) + "'"
        clock['stoptime'] = "'" + str(mkt.marketClearingTime + mkt.futureHorizon + timedelta(hours=mkt.intervalsToClear)) + "'"
        glmanip.write(dir_for_glm, model, clock, directives, modules, classes)

        print("Making GridLAB-D pre solve to get the forecasts")
        os.system("gridlabd solar.glm")

        os.chdir(path)

    def get_vertices_from_solar_forecasted_data(self, mkt):
        inflexible_building_folder = os.getcwd() + '/Solar_data/'
        csv_name = self.name + '.csv'
        filename = inflexible_building_folder+csv_name
        self.vertices = {}

        mkt_time = mkt.marketClearingTime
        with open(filename) as file:
            reader = csv.DictReader(file, fieldnames=["timestamp", "E"])
            for i in range(9):
                next(reader, None)
            for row in reader:
                timestamp_data = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S %Z')
                if mkt_time >= (mkt.marketClearingTime + mkt.futureHorizon + (mkt.intervalsToClear * mkt.intervalDuration)):
                    break
                if timestamp_data == mkt_time:
                    neutral_vertex_e = Vertex(marginal_price=float('0.0'), prod_cost=0.0, power=-1*float(row['E'])/1000)
                    vertices_val = [neutral_vertex_e]
                    vertices_type = [MeasurementType.PowerReal]
                    for type_energy, vert in enumerate(vertices_val):
                        iv = IntervalValue(self, mkt_time, mkt, MeasurementType.ActiveVertex, vert)
                        if str(vertices_type[type_energy]) in self.vertices:
                            self.vertices[str(vertices_type[type_energy])].append(iv)
                        else:
                            self.vertices[str(vertices_type[type_energy])] = []
                            self.vertices[str(vertices_type[type_energy])].append(iv)
                    mkt_time = mkt_time + mkt.intervalDuration

    def get_active_vertices_from_solar_pv_resource(self, mkt):
        # creates active vertices for the current time from the stored vertices
        # INPUTS:
        #
        # OUTPUTS:
        # vertices: the default minimum and maximum limit vertices

        self.activeVertices = {}
        for type_energy in self.vertices:
            mkt_time = mkt.marketClearingTime

            for iv in self.vertices[type_energy]:
                if  mkt_time >= (mkt.marketClearingTime + mkt.futureHorizon):
                    break
                if iv.timeInterval == mkt_time:
                    if type_energy in self.activeVertices:
                        self.activeVertices[type_energy].append(iv)
                    else:
                        self.activeVertices[type_energy]= []
                        self.activeVertices[type_energy].append(iv)
                    mkt_time = mkt_time + mkt.intervalDuration

        print("Read Active Vertices for {}".format(self.name))

    def update_dispatch(self, mkt, fed = None, helics_flag = bool(0)):

        elec_dispatched = self.scheduledPowers[str(MeasurementType.PowerReal)]*1000

        if helics_flag == True:
            key = "WSU_C_GLD_" + self.name + "_P_Out"
            try:
                pub = h.helicsFederateGetPublication(fed, key)
                status = h.helicsPublicationPublishDouble(pub, -1*elec_dispatched)
                print('Data {} Published to GLD {} via Helics -->'.format(-1*elec_dispatched, self.name))
            except:
                print('Publication was not registered')

        interval = mkt.marketClearingTime.strftime('%Y%m%dT%H%M%S')

        if len(self.record.keys()) == 0:
            self.record['TimeStamp']=[]
            self.record['TimeInterval']=[]
            self.record['Electricity Dispatched']=[]
        self.record['TimeStamp'].append(str(mkt.marketClearingTime))
        self.record['TimeInterval'].append(str(interval))
        self.record['Electricity Dispatched'].append(str(elec_dispatched))
        # line_new =  str(mkt.marketClearingTime) + "," + str(interval) + "," + str(-1*elec_dispatched) + " \n"
        # file_name = os.getcwd() + '/Outputs/' + self.name + '_output.csv'
        # try:
        #     with open(file_name, 'a') as f:
        #         f.writelines(line_new)
        # except:
        #     f = open(file_name, "w")
        #     f.writelines("TimeStamp,TimeInterval,Electricity Dispatched\n")
        #     f.writelines(line_new)
        #     # Add file location to index of outputs
        #     index_file_name = os.getcwd() + '/Outputs/pv_index.csv'
        #     try:
        #         with open(index_file_name, 'a') as indx_f:
        #             indx_f.writelines(self.name + ',' + file_name)
        #     except:
        #         indx_f = open(index_file_name, "w")
        #         indx_f.writelines("assetName, FileName\n")
        #         indx_f.writelines(self.name + ',' + file_name)
        #     indx_f.close()
        # f.close()

    def schedule_power(self, mkt):
        # Estimate stochastic generation from a solar
        # PV array as a function of time-of-day and a cloud-cover factor.
        # INPUTS:
        # obj - SolarPvResourceModel class object
        # tod - time of day
        # OUTPUTS:
        # p - calcalated maximum power production at this time of day
        # LOCAL:
        # h - hour (presumes 24-hour clock, local time)
        # *************************************************************************

        # Gather active time intervals
        tis = mkt.timeIntervals

        # Index through the active time intervals ti
        for ti in tis:
            # Production will be estimated from the time-of-day at the center of
            # the time interval.
            tod = ti.startTime + ti.duration/2  # a datetime

            # extract a fractional representation of the hour-of-day
            h = tod.hour
            m = tod.minute
            h = h + m / 60  # TOD stated as fractional hours

            # Estimate solar generation as a sinusoidal function of daylight hours.
            if h < 5.5 or h > 17.5:
                # The time is outside the time of solar production. Set power to zero.
                p = 0.0  # [avg.kW]

            else:
                # A sinusoidal function is used to forecast solar generation
                # during the normally sunny part of a day.
                p = 0.5 * (1 + math.cos((h - 12) * 2.0 * math.pi / 12))
                p = self.object.maximumPower[0] * p
                p = self.cloudFactor * p  # [avg.kW]

            # Check whether a scheduled power exists in the indexed time interval.
            iv = find_obj_by_ti(self.scheduledPowers[0], ti)
            if iv is None:
                # No scheduled power value is found in the indexed time interval.
                # Create and store one.
                iv = IntervalValue(self, ti, mkt, MeasurementType.ScheduledPower, p)
                # Append the scheduled power to the list of scheduled powers.
                self.scheduledPowers[0].append(iv)

            else:
                # A scheduled power already exists in the indexed time interval.
                # Simply reassign its value.
                iv.value = p  # [avg.kW]

            # Assign engagement schedule in the indexed time interval
            # NOTE: The assignment of engagement schedule, if used, will often be
            # assigned during the scheduling of power, not separately as
            # demonstrated here.

            # Check whether an engagement schedule exists in the indexed time interval
            iv = find_obj_by_ti(self.engagementSchedule[0], ti)

            # NOTE: this template assigns engagement value as true (i.e., engaged).
            val = True  # Asset is committed or engaged

            if iv is None:
                # No engagement schedule was found in the indexed time interval.
                # Create an interval value and assign its value.
                iv = IntervalValue(self, ti, mkt, MeasurementType.EngagementSchedule, val)  # an IntervalValue

                # Append the interval value to the list of active interval values
                self.engagementSchedule[0].append(iv)

            else:
                # An engagement schedule was found in the indexed time interval.
                # Simpy reassign its value.
                iv.value = val  # [$]

        # Remove any extra scheduled powers
        self.scheduledPowers[0] = [x for x in self.scheduledPowers[0] if x.timeInterval in tis]

        # Remove any extra engagement schedule values
        self.engagementSchedule[0] = [x for x in self.engagementSchedule[0] if x.timeInterval in tis]

# if __name__ == '__main__':
#     spvm = SolarPvResourceModel()
