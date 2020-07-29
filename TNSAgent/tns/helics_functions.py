from datetime import datetime, timedelta
import time
import helics as h
import random
import logging
import json
import sys
import os


# ################################  Create Broker ###############################
def create_broker(number_federates=3):
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    initstring = "--federates={} --name=mainbroker --loglevel=4".format(number_federates)
    broker = h.helicsCreateBroker("zmq", "", initstring)
    isconnected = h.helicsBrokerIsConnected(broker)

    if isconnected == 1:
        pass

    return broker


# ############################# register federate ###############################
def register_federate(json_filename):
    print('register_federate -->', json_filename)
    fed = h.helicsCreateValueFederateFromConfig(json_filename)
    status = h.helicsFederateRegisterInterfaces(fed, json_filename)
    federate_name = h.helicsFederateGetName(fed)[-1]
    print(" Federate {} has been registered".format(federate_name))
    pubkeys_count = h.helicsFederateGetPublicationCount(fed)
    subkeys_count = h.helicsFederateGetInputCount(fed)
    print(pubkeys_count)
    print(subkeys_count)
    # #####################   Reference to Publications and Subscription form index  #############################
    pubid = {}
    subid = {}
    for i in range(0, pubkeys_count):
        pubid["m{}".format(i)] = h.helicsFederateGetPublicationByIndex(fed, i)
        pub_type = h.helicsPublicationGetType(pubid["m{}".format(i)])
        pub_key = h.helicsPublicationGetKey(pubid["m{}".format(i)])
        print('Registered Publication ---> {} - Type {}'.format(pub_key, pub_type))
    for i in range(0, subkeys_count):
        subid["m{}".format(i)] = h.helicsFederateGetInputByIndex(fed, i)
        status = h.helicsInputSetDefaultString(subid["m{}".format(i)], 'default')
        sub_key = h.helicsSubscriptionGetKey(subid["m{}".format(i)])
        print('Registered Subscription ---> {}'.format(sub_key))

    return fed


def create_config_for_helics(source_node, target_bldg_model, gridlabd_assets=[], node_phase=0,
                             config_for_gridlabd=bool(False)):
    if len(source_node) > 5:
        source_node = source_node[0:5]  # source_node(1:5)

    # Shorten the name of the target node
    for i in range(len(target_bldg_model)):
        if len(target_bldg_model[i]) > 5:
            target_bldg_model[i] = target_bldg_model[i][0:5]  # target_node(1:5)

    config = {}
    gld_config = {}
    config['name'] = source_node
    config['loglevel'] = 7
    config['coreType'] = str('zmq')
    config['timeDelta'] = 1.0
    config['uninterruptible'] = bool('true')

    if config_for_gridlabd:
        gld_config['name'] = "GLD"
        gld_config['loglevel'] = 5
        gld_config['coreType'] = str('zmq')
        gld_config['period'] = 1.0
        gld_config['subscriptions'] = []
        gld_config['publications'] = []

    config['publications'] = []
    for i in range(len(target_bldg_model)):
        config['publications'].append({'key': str(source_node + '_' + target_bldg_model[i]),
                                       'type': str('string'),
                                       'global': bool('true'),
                                       })

    config['subscriptions'] = []
    for i in range(len(target_bldg_model)):
        config['subscriptions'].append({'key': str(target_bldg_model[i] + '_' + source_node),
                                        'type': str('string')})

    if config_for_gridlabd:
        phases = ['A', 'B', 'C']
        for asset in gridlabd_assets:
            if 'gld_info' not in asset.__dict__.keys(): # check if gld_info is a property in the asset. If not use defaults
                for phase in phases:
                    config['publications'].append({'key': str(source_node + '_GLD_' + asset.name + '_power_' + phase),
                                                   'global': bool('true'),
                                                   'type': str('complex')})
                    gld_config['subscriptions'].append({'key': str(source_node + '_GLD_' + asset.name + '_power_' + phase),
                                                        'type': str('complex'),
                                                        'unit': "VA",
                                                        'info': {"object": f"{asset.name}_helics",
                                                                 "property": f"constant_power_{phase}"}
                                                        })
            else:  # use info specified by asset to setup helics link
                if type(asset.gld_info['property']) is str:
                    properties = [asset.gld_info['property']]
                else:
                    properties = asset.gld_info['property']
                for prop in properties:
                    config['publications'].append({'key': str(source_node + '_GLD_' + asset.name + '_' + prop),
                                                   'global': bool('true'),
                                                   'type': asset.gld_info['type']})
                    # config['subscriptions'].append({'key': str('GLD_' + asset.name + '_' + source_node + '_' + prop),
                    #                                 'type': asset.gld_info['type']})

                    gld_config['subscriptions'].append({'key': str(source_node + '_GLD_' + asset.name + '_' + prop),
                                                        'type': asset.gld_info['type'],
                                                        'unit': "VA",
                                                        'info': {"object": asset.gld_info['object'],
                                                                 "property": prop}})
                    # gld_config['publications'].append({'key': str('GLD_' + asset.name + '_' + source_node + '_' + prop),
                    #                                     'type': asset.gld_info['type'],
                    #                                     'unit': "VA",
                    #                                     'global': bool(True),
                    #                                     'info': {"object": asset.gld_info['object'],
                    #                                              "property": prop}})


    if config_for_gridlabd:
        gld_json_filename = 'network/gld_config.json'
        gld_json_file = json.dumps(gld_config, indent=4, separators=(',', ': '))
        with open(gld_json_filename, 'w') as fp:
            print(gld_json_file, file=fp)

    json_filename = source_node + '_config.json'
    json_file = json.dumps(config, indent=4, separators=(',', ': '))
    with open(json_filename, 'w') as fp:
        print(json_file, file=fp)

    return json_filename


def destroy_federate(fed):
    h.helicsFederateFinalize(fed)
    h.helicsFederateFree(fed)
    h.helicsCloseLibrary()
