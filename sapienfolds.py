from inertial_units import *
from lie_group_utils import SO3, SE3_2
import preintegration_utils as preint 
torch.set_default_dtype(torch.float64)

'''

Human motion estimation on Lie groups using IMU measurements # 2017 IEEE/RSJ International Conference
Vladimir Joukov∗ , Josip Ćesić‡ , Kevin Westermann∗ , Ivan Marković‡ , Dana Kulić∗ and Ivan Petrović‡

diff eqs:
X_{k+1} = X_{k} exp_{ \\omega_k + n_k}, n_k = 0 for now. 
(\\omega_k^i)^T = [T \\omega_k^i + 0.5 T^2 \\alpha_k^i , T \\alpha_k^i, 0]

Z_{k+1} = h(X_{k+1} exp^{m_{k+1}})

'''
class Joukov_et_al_approch:
	def __init__(self):
		self.joints = []
		self.Kalman_Gain = lambda : None

		self.R = lambda x: torch.tensor( # basis vector of SO(3)
			[ [0, -x[2], x[1] ],
			[ x[3], 0, -x[0] ],
			[-x[1], x[0], x[0]]
		 ] )
		# 
	
	def measurement_update(data, device):
		# to do:
		# get device information, regarding placement on body, get distance from device to {all joints}
		# 
		# 
		K = lambda i_joint, s_sensor: None
		#gyro update
		h_gyro = None
		#accel update
	def exp_SO3(self, x):
		# taylor expanded exp map of SO(3)
		norm = torch.norm(x)
		so3 = torch.matmul( torch.cos( norm) , torch.eye(3) )
		so3 += torch.matmul( 1 - torch.cos(norm) , torch.matmul( x, torch.transpose(x) ) / np.matmul(norm, norm) )

		so3 += torch.matmul (torch.sin(norm), self.R(x) / norm)
		return so3 

def aquire_data(port):
	device = IMU_Device(device_id = f'testing_{port}')
	device.connect(server_port = port)
	device.listen(save_file = "testing_data_saving")
def main():
	active_ports = range(1234, 1238+1) #  
	for port in active_ports:
		exec ( f'thread_{port} = threading.Thread( target = aquire_data, args = ({port}, ) , daemon = True )' )
		exec ( f" thread_{port}.start() " )

if __name__ == '__main__':
	main()