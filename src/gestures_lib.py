

## TODO: Move gestures detection lib from leapmotionlistener.py

import os
import gdown

import settings


class GesturesDetectionClass():
    def __init__(self):
        pass

    @staticmethod
    def download_networks_gdrive():
        # get one dir above
        NETWORKS_PATH = '/'.join((settings.NETWORK_PATH).split('/')[:-2])
        gdown.download_folder(settings.NETWORKS_DRIVE_URL, output=NETWORKS_PATH)


    def change_current_network(self, network=None):
        ''' Switches learned file
        '''
        pass

    @staticmethod
    def get_networks():
        ''' Looks at the settings.NETWORK_PATH folder and returns every file with extension *.pkl
        '''
        networks = []
        for file in os.listdir(settings.NETWORK_PATH):
            if file.endswith(".pkl"):
                networks.append(file)
        return networks


class Network():
    def __init__(self, file):
        self.name = file
