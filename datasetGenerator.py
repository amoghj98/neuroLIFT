#!/usr/bin/python

import numpy as np
import torch as th

import h5py


class datasetGenerator(object):
	def __init__(self, n_samples=500, a_max=1.25214, v_max=17.53172, x_min=-1.2, x_max=1.2, y_min=0.4, y_max=1.0, z_min=0.0, z_max=1.2):
		self.nSamples = n_samples
		self.a_max = a_max
		self.v_max = v_max
		self.agent_posRange = np.array([[x_min, y_min, z_min], [x_max, y_max, z_max]])
		self.obst_posRange = np.array([[-1.2, 0.4, -0.1], [1.2, 0.9, 0.1]])
		self.agent_velRange = np.array([[-1.2, -0.7, -1.2], [1.2, 0.7, 1.2]])
		self.obst_velRange = np.array([[-0.6, -0.05, -0.01], [0.6, 0.05, 0.01]])
		self.timRange = np.array([0.0, 1.5])
		self.agent_position = np.zeros((self.nSamples, 3))
		self.obst_position = np.zeros((self.nSamples, 3))
		self.agent_velocity = np.zeros((self.nSamples, 3))
		self.obst_velocity = np.zeros((self.nSamples, 3))
		self.t_to_col = np.random.uniform(self.timRange[0], self.timRange[1], self.nSamples)[:, np.newaxis]
		# setup initial poses and velocities
		for i in range(3):
			self.agent_position[:, i] = np.random.uniform(self.agent_posRange[0, i], self.agent_posRange[1, i], self.agent_position.shape[0])
			self.agent_velocity[:, i] = np.random.uniform(self.agent_velRange[0, i], self.agent_velRange[1, i], self.agent_velocity.shape[0])
			self.obst_position[:, i] = np.random.uniform(self.obst_posRange[0, i], self.obst_posRange[1, i], self.obst_position.shape[0])
			self.obst_velocity[:, i] = np.random.uniform(self.obst_velRange[0, i], self.obst_velRange[1, i], self.obst_velocity.shape[0])
		# compute distance to target and net agent_velocity
		ob_pos = (self.obst_position + (self.obst_velocity * self.t_to_col))
		self.d = np.sqrt(((ob_pos - self.agent_position)**2).sum(axis=1))[:, np.newaxis]	# shape: [n, 1]
		self.v = np.sqrt((self.agent_velocity**2).sum(axis=1))[:, np.newaxis]				# shape: [n, 1]

	def debug(self):
		for k in self.__dict__.keys():
			if type(self.__dict__[k]) is np.ndarray:
				print(f'{k}: {self.__dict__[k].shape}')
				print(f'{k}: {self.__dict__[k]}')

	def kinematic_feasibility(self):
		self.a = ((2 * (self.d - (self.v * self.t_to_col))) / self.t_to_col**2)
		return self.a < self.a_max
	
	def agent_velocity_feasibility(self):
		vf = self.v + (self.a * self.t_to_col)
		return vf < self.v_max
	
	def temporal_feasibility(self):
		return self.d > (self.v_max * self.t_to_col)
	
	def get_labels(self):
		self.labels = np.logical_and(self.kinematic_feasibility(), self.agent_velocity_feasibility(), self.temporal_feasibility())
		return 0

	def write_dataset(self):
		self.get_labels()
		prompts = ["Is it feasible to navigate through the obstacle?", "the drone can negotiate the obstruction successfully?", "the drone can effectively traverse the barrier in place?", "it is possible to find a way past the hurdle ahead?", "the drone can bypass the roadblock with ease?", "there is a clear strategy to navigate this challenge?"]
		prompt_set = np.array([p.encode('ascii', 'ignore') for p in prompts])
		datasetPath = '/scratch/gautschi/joshi157/finetuningDatasets/'
		filename = 'neuroLIFT.hdf5'
		file = datasetPath + filename
		f = h5py.File(file, 'a')
		f.create_dataset("prompts", data=prompt_set, dtype=prompt_set.dtype, chunks=True)
		f.create_dataset("agent_pos", data=self.agent_position, dtype=self.agent_position.dtype, chunks=True)
		f.create_dataset("agent_vel", data=self.agent_velocity, dtype=self.agent_velocity.dtype, chunks=True)
		f.create_dataset("obst_pos", data=self.obst_position, dtype=self.obst_position.dtype, chunks=True)
		f.create_dataset("obst_vel", data=self.obst_velocity, dtype=self.obst_velocity.dtype, chunks=True)
		f.create_dataset("t_to_col", data=self.t_to_col, dtype=self.t_to_col.dtype, chunks=True)
		f.create_dataset("labels", data=self.labels, dtype=self.labels.dtype, chunks=True)
		f.close()
	

if __name__ == "__main__":
	d = datasetGenerator(n_samples=5000)
	# d.get_labels()
	# d.debug()
	d.write_dataset()
	print(np.count_nonzero(d.labels == True))