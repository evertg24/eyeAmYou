# for jupyter/colab ================================================================================================================================================== #

#%matplotlib notebook
#%matplotlib inline
import os
import sys
import torch
import threading 
import pandas as pd 
import datetime

# for jupyter/colab ================================================================================================================================================== #


import socket 
import time
import urllib.request
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
import numpy as np
from tqdm.notebook import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80



class IMU_Device: #
	'''
	add data saving feature,
	'''

	def __init__(self, _id = 'device_' + str(np.random.randomint()) ):
		
		self.device_id = _id
		self.calibration_values = []

	def OTA_update(self, file):
		# update sock addy/port
		# update calibration values
		# adding hardware interrrupts maybe, not really needed
		pass #to do 
	def listen(self, save_file, ζ = 1e3):

		today = datetime.datetime.today().strftime('%m_%d_%Y')

		save_folder = f'IMU_Data/{self.device_id}'

		clean_stack = lambda : pd.DataFrame(
			columns = ['ax', 'ay', 'az', 'wx', 'wy', 'wz','bx', 'by', 'bz', 'T']
			)
		stack_ = clean_stack()
		i = 0

		while True:
			if (i % ζ) == 0: # To-Do : start new thread to save data, while still aquiring data 
				stack_.to_csv(save_folder + f'/{save_file}.csv')
				stack_ = clean_stack()
				i+=1

			x, _ = self.sock.recvfrom(100)
			data = [float(ξ) for ξ in x.decode().split('e')]
			stack_.loc[ data[0]] = data[1::]
			i+=1;


	def connect(self, # assuming something, iforget
	 external_ip = urllib.request.urlopen('https://ident.me').read().decode('utf8'),
	 server_port = 1224, *args): # Async-UDP connection 

		self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # clear addy so resuable

		server_address = str(external_ip)
		server = (server_address, server_port)

		print("server = , ", server)
		self.sock.bind(server)
		print(f"Server built @ {server} .\n\n")



			

class Dasein:
	'''
	load patient obj file. 
	vert.shape = FloatTensor(V, 3),
	face.shape = LongTensors [vert_index, normals_indx, texture_indx]
	'''
	def __init__(self, file, normalize=True, *args):
		self.name = str(file)[:-4]
		self.TTM = np.array([0,0,0,0,0,0]) #
		self.TM = np.array([0,0,0,0,0,0])
		self.tempreture = 0; # celcius
		self.magnetic_field = np.array([0,0,0])
		self.g_μv = np.array([[1,0,0], [0,1,0], [0,0,1]])

		if not isinstance(file, str): # except , huh ? 
			self.mesh = file

		else:
			self.verts, self.faces, self.aux = load_obj(file)
			self.faces_idx, self.verts = self.faces.verts_idx.to( __device__ ), self.verts.to(__device__)
			if normalize:
				self.verts = self.verts - self.verts.mean(0)
				self.verts = self.verts / max(self.verts.abs().max(0)[0])
			self.mesh = Meshes( verts = [self.verts,] , faces = [self.faces_idx,] )



	def plot_pointcloud( self, N = int(1e3), *args):
		x, y, z = sample_points_from_meshes(self.mesh, N).clone().detach().cpu().squeeze().unbind(1)
		fig = plt.figure(figsize = (5,5))
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter3D(x, y, z)
		ax.set_xlabel("x"); ax.set_ylabel('y'); ax.set_zlabel('z') 
		ax.set_title(self.name)
		ax.view_init(190, 30)

		plt.show()
		#plt.savefig('cow' + self.name )
	def save(self, ignore_this_shit): # ignore example
		# Fetch the verts and faces of the final predicted mesh
		final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

		# Scale normalize back to the original target size
		final_verts = final_verts * scale + center

		# Store the predicted mesh using save_obj
		final_obj = 'final_model.obj'
		save_obj(final_obj, final_verts, final_faces)


def main():
	# testing simulated IMU data
	N_devices = 3
	for i in range(1000):
		TTM = [ np.random.random((3,)) for dimension in range(3) ]


def main_1():
	global __device__; 

	if torch.cuda.is_available():
		__device__ = torch.device('cuda:0')
	else:
		__device__ = torch.device('cpu')
	print(f"running calcs on :\t {__device__}")
	root_dir = '/nas/longleaf/home/evertg24/Inertial_Units'
	example = root_dir + '/SPRING_MALE/SPRING_MALE/SPRING0001.obj'

	#example = r'/SPRING_MALE/mesh/SPRING0014.obj'
	not_evert = Dasein(file = example)
	refrence_obj = Dasein(file = ico_sphere(4, __device__), )
	#not_evert.plot_pointcloud()
	#refrence_obj.plot_pointcloud()
	t1 = threading.Thread(target = not_evert.plot_pointcloud, args = () , daemon = True)
	t2 = threading.Thread(target = refrence_obj.plot_pointcloud, args = () , daemon = True)
	t1.start(); t2.start()
	#t1.join(); t2.join()
	plt.close('all')




if __name__ == '__main__':
	main()