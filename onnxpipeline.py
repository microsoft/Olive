import os
import os.path as osp
import docker
import config

class Onnxpip:
    def __init__(self, directory):
        #if not directory: raise Exception('no target directory')
        if directory[0] == '/':
            self.path = directory
        else:
            self.path = osp.join(os.getcwd(), directory)
        self.client = docker.from_env()

    def convert_model(self):
        pass

    def create_input(self):
        img_name = (config.CONTAINER_NAME + 
            config.FUNC_NAME['create_input'] + ':latest')
    
        mount_model = osp.join(config.MOUNT_PATH, config.CONVERTED_MODEL)
        self.client.containers.run(image=img_name, 
            command='--model ' + mount_model, 
            volumes={self.path: {'bind': config.MOUNT_PATH, 'mode': 'rw'}})