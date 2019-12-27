from datetime import datetime, timedelta
import time
import helics as h
import random
import logging
import json
import sys
import os

#################################  Create Broker ###############################
def create_broker():
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    initstring = "--federates=3 --name=mainbroker"
    broker = h.helicsCreateBroker("zmq", "", initstring)
    isconnected = h.helicsBrokerIsConnected(broker)

    if isconnected == 1:
        pass

    return broker
############################## register federate ###############################
def register_federate(json_filename):
    print('register_federate -->',json_filename)
    fed = h.helicsCreateValueFederateFromConfig(json_filename)
    status = h.helicsFederateRegisterInterfaces(fed, json_filename)
    federate_name = h.helicsFederateGetName(fed)[-1]
    print(" Federate {} has been registered".format(federate_name))
    pubkeys_count = h.helicsFederateGetPublicationCount(fed)
    subkeys_count = h.helicsFederateGetInputCount(fed)
    print(pubkeys_count)
    print(subkeys_count)
    ######################   Reference to Publications and Subscription form index  #############################
    pubid = {}
    subid = {}
    for i in range(0,pubkeys_count):
        pubid["m{}".format(i)] = h.helicsFederateGetPublicationByIndex(fed, i)
        pub_type = h.helicsPublicationGetType(pubid["m{}".format(i)])
        pub_key = h.helicsPublicationGetKey(pubid["m{}".format(i)])
        print( 'Registered Publication ---> {} - Type {}'.format(pub_key, pub_type))
    for i in range(0,subkeys_count):
        subid["m{}".format(i)] = h.helicsFederateGetInputByIndex(fed, i)
        status = h.helicsInputSetDefaultString(subid["m{}".format(i)], 'default')
        sub_key = h.helicsSubscriptionGetKey(subid["m{}".format(i)])
        print( 'Registered Subscription ---> {}'.format(sub_key))

    return fed

def create_config_for_helics(source_node,target_node, gridlabd_nodes = [], node_phase = 0, config_for_gridlabd = bool(False)):
    if len(source_node) > 5:
        source_node = source_node[0:5]  # source_node(1:5)

    # Shorten the name of the target node
    for i in range(len(target_node)):
        if len(target_node[i]) > 5:
            target_node[i] = target_node[i][0:5]  # target_node(1:5)

    config = {}
    config['name'] = source_node
    config['loglevel'] = 7
    config['coreType'] = str('zmq')
    config['timeDelta'] = 1.0
    config['uninterruptible'] = bool('true')

    config['publications'] = []
    for i in range(len(target_node)):
        config['publications'].append({'global': bool('true'),
                                       'key': str(source_node+'_'+target_node[i]),
                                       'type': str('string')})

    config['subscriptions'] = []
    for i in range(len(target_node)):
        config['subscriptions'].append({'required': bool('true'),
                                       'key': str(target_node[i]+'_'+source_node),
                                       'type': str('string')})

    if config_for_gridlabd == True:
        phases = ['A', 'B', 'C']
        for i in range(11):
            for j in range(len(phases)):
                config['publications'].append({'global': bool('true'),
                                           'key': str(source_node + '_GLD_' + gridlabd_nodes[i+2] + '_power_' + phases[j]),
                                           'type': str('complex')})

    json_filename = source_node+'_config.json'
    json_file = json.dumps(config, indent=4, separators = (',',': '))
    fp = open(json_filename, 'w')
    print(json_file, file = fp)
    fp.close()

    return json_filename

def destroy_federate(fed):
    status = h.helicsFederateFinalize(fed)
    h.helicsFederateFree(fed)
    h.helicsCloseLibrary()
