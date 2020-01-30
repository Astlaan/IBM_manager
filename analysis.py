from inspect import getsourcefile
from qiskit import IBMQ, compiler
import os.path as path, sys
import os
import argparse
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
from simulation import SimManager
from size_reducer import get_file_list
import pickle
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from qiskit.tools.events import TextProgressBar
from joblib import Parallel, delayed
from tqdm import tqdm

# from pandarallel import pandarallel
# pandarallel.initialize()

def load_job_result(absfilepath):
	with open(absfilepath, 'rb') as the_file:
		data = pickle.load(the_file)
	return (os.path.split(absfilepath)[1], data)

def get_file_list(input_path, ext):
	return [f for f in os.listdir(input_path) if f.endswith(ext)]
	
def parallelize_dataframe(df, func, n_cores=cpu_count()):
	df_split = np.array_split(df, n_cores)
	pool = Pool(n_cores)
	df = pd.concat(pool.map(func, df_split))
	pool.close()
	pool.join()
	return df

class Analysis:
	def __init__(self, res_dir = None, sim_dir=None, load_file=None, name_as_index=True, dtype = None):
		if res_dir and sim_dir:
			self.res_dir = res_dir
			self.sim_dir = sim_dir
			self.variability = False
			self.df = pd.DataFrame(columns=['circ_name','real_results','correct_result', 'size', 'depth'])
			if name_as_index:
				self.df.set_index('circ_name')
				self.df.drop(['circ_name'], axis=1, inplace=True)
		elif load_file:
			self.load(load_file, dtype = dtype)
		else:
			print('\nANALYSIS CLASS ARGUMENT ERRORS....\n')


	def load_all_job_results(self, kind):
		ext = '.pkl'
		if kind == 'sim':
			dir = self.sim_dir
		elif kind == 'real':
			dir = self.res_dir
		filelist = get_file_list(dir, ext)
		data = []

		for file in filelist:
			data.append(load_job_result(os.path.join(dir, file)))

		return dict(data)

	def extract_data(self, real = False, sim = False):
		if real:
			real_results = self.load_all_job_results('real')
		if sim:
			sim_results = self.load_all_job_results('sim')
		
		if real:
			for name, result in real_results.items():
				# print(name, result)
				self.df.loc[name, 'real_results'] = [result.get_counts()] #using [] encapsulation was a workaround for pandas not accepting dictionaries
		if sim: 
			for name, result in sim_results.items():
				# print(name, result)
				self.df.loc[name, 'correct_result'] = [result.get_counts()]

		return sim_results.items()


	
	def extract_parameters_cols(self):
		circ_names = set(self.df.index.values)
		for circ in circ_names:
			self.df.loc[circ,"f1"] = int(circ.split("f1")[1].split("_")[0])
			self.df.loc[circ,"g"] = int(circ.split("g")[1].split("_")[0])
			self.df.loc[circ,"l"] = int(circ.split("l")[1].split("_")[0])
			self.df.loc[circ,"f2"] = int(circ.split("f2")[1][0:1])


		return sim_results.items()



	def get_depth_and_size(self, dir, transpile = True, backend_name = None):
		# def _get_size_and_depth(self, row):
		#     name = row.name
		#     circ = QuantumCircuit.from_qasm_file(os.path.join(self.res_dir, name))
		#     row['depth'] = circ.depth()
		#     row['size'] = circ.size()
		# self.df.parallel_apply(_get_size_and_depth, axis=1)

		# def _get_size_and_depth(slice):
		#     for row in slice.iterrows():
		#         name = row.name
		#         circ = QuantumCircuit.from_qasm(os.path.join(self.res_dir, name))
		#         row['depth'] = circ.depth()
		#         row['size'] = circ.size()
		# self.df = parallelize_dataframe(self.df, _get_size_and_depth)
		if backend_name:
			IBMQ.load_accounts()
			backend = IBMQ.backends(filters=lambda x: x.name() == backend_name)[0]
		file_list = get_file_list(dir, '.qasm')
		file_list_abs = [ os.path.join(dir, file) for file in file_list ]
		pool = Pool()
		print('==Creating circuit list==')
		# circ_list = pool.map(QuantumCircuit.from_qasm_file, file_list_abs)
		circ_list = Parallel(n_jobs=cpu_count())(delayed(QuantumCircuit.from_qasm_file)(file) for file in file_list_abs)
		# pool.close()
		# pool.join()
		if transpile:
			print('==Transpiling!==')
			TextProgressBar()
			circ_list = compiler.transpile(circ_list, backend)
		file_list_pkl = [ os.path.splitext(file)[0]+'.pkl' for file in file_list ]

		for i in range(len(file_list_pkl)):
			self.df.loc[file_list_pkl[i], 'depth'] = circ_list[i].depth()
			self.df.loc[file_list_pkl[i], 'size'] = circ_list[i].size()
 
		return


	def _middle(self, row):
		# if 'middle' in row['circ_name']:
		#     print(row['circ_name'])
		print (self.df.index(row))
			# row['middle'] = True
		# else:
			# row['middle'] = False

	def calculate_prob_success(self):
		def _pd_row_prob_success(row):
			# the [0] everywhere is due to to the encapsulation used in self.extract_data
			# if len(row['correct_result'][0].items()) == 1:
			if type(row['correct_result']) is not list or type(row['real_results']) is not list:
				return
			# correct = next(iter( row['correct_result'][0] ))

			correct = max(row['correct_result'][0], key = lambda k: row['correct_result'][0][k] )
			try:
				correct_count = row['real_results'][0][correct] 
			except:
				correct_count = 0
			prob_succ = correct_count/sum(row['real_results'][0].values()) * 100
			return prob_succ

		prob_succ_col = self.df.apply(_pd_row_prob_success, axis=1)
		self.df['prob_succ'] = prob_succ_col
		self.sort_prob_succ()
	
	def sort_prob_succ(self, by = 'prob_succ'):
		self.df.sort_values(by = by, ascending= False)

	def save(self, filepath):
		dir = os.path.dirname(filepath)
		if dir and not os.path.exists(dir):
			os.makedirs(os.path.dirname(dir))
		self.df.to_csv(filepath, index=True, header=True)
		print('===\nSaved to: ' + filepath + '\n===')
		# with open(file, 'wb') as the_file:
		# 	pickle.dump(self.df, the_file, protocol=2)

	def load(self, filepath, dtype):
		self.df = pd.read_csv(filepath, index_col=0, dtype = dtype)
		# with open(file, 'rb') as the_file:
		# 	self.df = pickle.load(the_file)
		# print('===\nLoaded: ' + file + '\n===')

	def success_plot(self, output = None):
		from mpl_toolkits.mplot3d import Axes3D
		threedee = plt.figure()
		# threedee = plt.figure().gca(projection='3d')
		ax = threedee.add_subplot(111, projection='3d')
		ax.scatter(self.df['depth'].tolist(), self.df['size'].tolist(), self.df['prob_succ'].tolist())
		ax.set_xlabel('depth')
		ax.set_ylabel('size')
		ax.set_zlabel('prob. sucess')
		if output:
			os.makedirs(os.path.dirname(output))
			plt.savefig(output)
		else:
			plt.savefig('success_plot.png')
	
	def print(self):
		if self.variability == True:
			print(analysis.df['path', 'dist', 'init', 'control', 'size', 'depth', 'prob_succ'])
		else:
			print(analysis.df)
		

		


if __name__== "__main__":

	parser = argparse.ArgumentParser(description='Analysis framework (panda powered)')
	parser.add_argument('--res_indir', type=str, help='Input dir for circuits')
	parser.add_argument('--sim_indir', type=str, help='Input dir for circuits')

	parser.add_argument('-v',  action="store_true", default=False, help='Variability study')
	parser.add_argument('-l',  type=str, help='Load Dataframe') 

	# parser.add_argument('outdir', type=str, help='Output dir for results')
	# parser.add_argument('-s', type=int, default=14, help='Number of qubits') #NOT IMPLEMENTED
	args = parser.parse_args()

	curdir = os.getcwd()

	# if not os.path.exists(output_path):
		# os.makedirs(output_path)

	if not args.l:
		res_indir = os.path.join(curdir, args.res_indir)
		sim_indir = os.path.join(curdir, args.sim_indir)
		analysis = Analysis(res_dir = res_indir, sim_dir = sim_indir)
		analysis.extract_data(real = True, sim = True)
		analysis.calculate_prob_success()
	else:
		analysis = Analysis(load_file = args.l)
		print(analysis.df)

	if args.v:
		analysis.calculate_prob_success()
		analysis.variability = True
		
		for index, row in tqdm(analysis.df.iterrows()):
			path = index.split('_i')[0]
			len_path = index.count('_') - 1
			distance = len_path - 1
			initial_state = index.split('s')[1].split('_')[0]# #after the 's' but before the next _ 
			control_index = int(index.split('i')[1].split('_')[0]) #after the 'i' but before the next _ 
			meas_interm = index.split('m')[1].split('.')[0] #after the 'i' but before the next _ 
			
			analysis.df.loc[index, 'path'] = path
			analysis.df.loc[index, 'dist'] = distance
			analysis.df.loc[index, 'init'] = initial_state
			analysis.df.loc[index, 'cntr'] = int(path.split('_')[control_index])
			analysis.df.loc[index, 'cntr_frac'] = control_index/(distance-1) * 100
			analysis.df.loc[index, 'meas_interm'] = meas_interm
		analysis.df = analysis.df.astype({'cntr' : int, 'dist' : int })
		analysis.get_depth_and_size("variability_v2", backend_name = 'ibmq_16_melbourne')

			# analysis.additional =  {'worst_state':}

		# middle_better_count = 0
		# for index, row in analysis.df.iterrows():
		#     if 'middle' in index:
		#         name = index.replace('middle', '')
		#         if row['prob_succ'] > analysis.df.loc[name, 'prob_succ']:
		#             middle_better_count += 1
		# print('\n\nMiddle routing was better ', middle_better_count/(len(analysis.df.index)/2)*100, '% of the times\n\n')

			




	



	import code
	code.interact(local=locals())
