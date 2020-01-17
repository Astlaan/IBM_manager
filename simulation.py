import argparse
import os
import glob
from qiskit import QuantumCircuit, execute, Aer
import pickle
from qiskit import IBMQ
from qiskit.providers import JobStatus
from copy import deepcopy
from qiskit.tools.monitor import job_monitor
from qiskit import compiler
from qiskit.tools.events import TextProgressBar
import time
import multiprocessing as mp
from qiskit.transpiler.transpile_config import TranspileConfig
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm
import traceback

class SimManager:
	def __init__(self, input_path, output_path, backend, circs_per_job, optim, number_of_jobs):
		self.input_path = input_path
		self.output_path = output_path
		self.circs_per_job = circs_per_job
		self.number_of_jobs = number_of_jobs
		self.optim = optim
		if not os.path.exists(output_path):
			os.makedirs(output_path)
		self.backend = backend
		if backend.name() == 'ibmq_qasm_simulator' or backend.name() == 'statevector_simulator' :
			self.shots = 1024
		else: 
			self.shots = 1024


	def save_circuit_result(self, job_result, index):
		job_result = deepcopy(job_result)
		filename = job_result.results[index].header.name

		while True:
			if filename in self.output_path:
				file += '_A'
			else:
				break

		job_result.results = [ job_result.results[index] ]
		with open( os.path.join(self.output_path, os.path.splitext(filename)[0] + '.pkl'  ) , 'wb') as the_file:
			pickle.dump(job_result, the_file, protocol = 0)
		print('SAVED ' + os.path.splitext(filename)[0] + '.pkl')

	def save_job_result (self, job_result):
		number_of_circuits = len(job_result.results)

		for i in range(number_of_circuits):
			self.save_circuit_result(job_result, i)

	def create_circ(self, absfilepath):
		circ = QuantumCircuit.from_qasm_file(absfilepath)
		circ.name = os.path.basename(absfilepath)
		return circ

	def create_circ_list(self, filenamelist):
		circ_list = []
		print('CREATING CIRCUIT LIST')
		pool = mp.Pool(mp.cpu_count())
		circ_list = pool.map(self.create_circ, [os.path.join(self.input_path, file) for file in filenamelist])
		pool.close()
		pool.join()

		# circ_list = Parallel(n_jobs=mp.cpu_count(), verbose = 49)(delayed(self.create_circ)(os.path.join(self.input_path, file)) for file in filenamelist)
		print('FINISHED CREATING CIRC LIST')

		return circ_list

	# def create_circ_list_from_qasm_dict(self, name_qasm_dict):
	#     def create_circ_from_qasm(self, name_qasm_item):
	#         name, qasm = name_qasm_item
	#         circ = QuantumCircuit.from_qasm_str(qasm)
	#         circ.name = name
	#         return circ
		
	#     circ_list = []

	#     pool = mp.Pool(mp.cpu_count())
	#     circ_list = pool.map(create_circ_from_qasm, name_qasm_dict.items())
	#     pool.close()
	#     pool.join()

	#     return circ_list
	
	
	def get_unrun_files(self, transpile_only = False):
		input_files =  [f for f in os.listdir(self.input_path) if f.endswith(".qasm")]
		# output_files = [f for f in os.listdir(self.output_path) if f.endswith(".pkl")]
		# replace .pkl -> .qasm in output files so comparison is possible with input files

		if transpile_only:
			out_ext = '.qasm'
		else:
			out_ext = '.pkl'
		
		output_files = [os.path.splitext(f)[0] + '.qasm' for f in os.listdir(self.output_path) if f.endswith(out_ext)]
		
		input_files = list(set(input_files).difference(output_files))
		return input_files

	def cancel_all_running_jobs(self):
		jobs = self.backend.jobs()
		for job in reversed(jobs[-self.number_of_jobs:-1]):
			try:
				job.cancel()
			except:
				continue
	def remove_failed_jobs(self):
		for job in self.current_jobs:
			try:
				if job.status() == JobStatus.ERROR:
					print('ERR: A Job has failed')
					self.current_jobs.remove(job)
			except:
				continue

	def save_qasm(self, circ):
		output = circ.qasm()
		with open(os.path.join(self.output_path, circ.name), 'w') as the_file:
			the_file.write(output)
	

	def transpile(self):
		input_files = self.get_unrun_files(transpile_only = True)
		print('\n ============= TRANSPILATION ONLY ================')
		print('\n RESUMING \n')
		print(len(input_files), ' circuits left:')
		print(input_files)
		circ_list = self.create_circ_list(input_files)
		print('Circuits generated. Transpiling...')
		TextProgressBar()
		circ_list = compiler.transpile(circ_list, backend=self.backend, optimization_level = self.optim)

		for circ in circ_list:
			self.save_qasm(circ)

		# pool = mp.Pool(mp.cpu_count(), maxtasksperchild=1)

		# config = TranspileConfig(self.optim, backend=self.backend)
		# circ_list = pool.map_async(parallel_transpiler_wrapper, list(product(circ_list, config)), callback = save_qasm)
		# print('pool map async started')
		# pool.close()
		# pool.join()




	def start(self):
		
		if self.backend == 'ibmq_16_melbourne':
			self.cancel_all_running_jobs()
		#print(os.getcwd())

		input_files = self.get_unrun_files()
		print('=================== SIMULATING (FULL) =====================')
		print('\n RESUMING \n')
		print(len(input_files), ' circuits left:')
		print(input_files)

		circ_list = self.create_circ_list(input_files)
		#Divide circuits in jobs
		print('Creating jobs...')
		job_list = [circ_list[x:x+self.circs_per_job] for x in range(0, len(circ_list), self.circs_per_job)]
		#Divide the jobs in chunks of five, so that they fit the credits (3 creds per job)
		print('Creating job chunks...')
		job_chunks = [job_list[x:x+self.number_of_jobs] for x in range(0, len(job_list), self.number_of_jobs)]
		
		for chunk in job_chunks:
			self.current_jobs = []
			print('\n SIMULATING: ', chunk)
			for job in chunk:
				print('NEW JOB')
				# circ_list = self.create_circ_list(job)
				# print('Circuits created from files: ', [circ.name for circ in circ_list])
				initial_time = time.time()
				job_run = execute(job, self.backend, shots=self.shots, max_credits=3, optimization_level = self.optim )
				print('JOB REQUEST SENT! took ', (time.time() - initial_time)/60.0, 'mins ', job_run.status() )
				self.current_jobs.append(job_run)
			self.remove_failed_jobs()
			# job_monitor(self.current_jobs[-1])
			results = []
			for job_run in self.current_jobs:
				try:
					result = job_run.result()
					results.append(result)
				except:
					try: 
						print(job_run.status())
					except Exception: 
						traceback.print_exc()
			# results = [job_run.result() for job_run in self.current_jobs]
			for job_result in results:
				self.save_job_result(job_result)

	# def start(self):
		
	#     if self.backend == 'ibmq_16_melbourne':
	#         self.cancel_all_running_jobs()
	#     #print(os.getcwd())

	#     input_files = self.get_unrun_files()
	#     print('=================== SIMULATING (FULL) =====================')
	#     print('\n RESUMING \n')
	#     print(len(input_files), ' circuits left:')
	#     print(input_files)

	#     #Divide circuits in jobs
	#     job_list = [input_files[x:x+self.circs_per_job] for x in range(0, len(input_files), self.circs_per_job)]
	#     #Divide the jobs in chunks of five, so that they fit the credits (3 creds per job)
	#     job_chunks = [job_list[x:x+5] for x in range(0, len(job_list), 5)]
		
	#     for chunk in job_chunks:
	#         self.current_jobs = []
	#         print('\n SIMULATING: ', chunk)
	#         for job in chunk:
	#             print('NEW JOB')
	#             circ_list = self.create_circ_list(job)
	#             print('Circuits created from files: ', [circ.name for circ in circ_list])
	#             initial_time = time.time()
	#             job_run = execute(circ_list, self.backend, shots=self.shots, max_credits=3, optimization_level = self.optim )
	#             print('JOB REQUEST SENT! took ', (time.time() - initial_time)/60.0, 'mins ', job_run.status() )
	#             self.current_jobs.append(job_run)
	#         self.remove_failed_jobs()
	#         job_monitor(self.current_jobs[-1])
	#         results = [job_run.result() for job_run in self.current_jobs]
	#         for job_result in results:
	#             self.save_job_result(job_result)


if __name__== "__main__":

	parser = argparse.ArgumentParser(description='Check validity of files in a given directory')
	parser.add_argument('indir', type=str, help='Input dir for circuits')
	parser.add_argument('outdir', type=str, help='Output dir for results')
	parser.add_argument('backend', type=str, help='Backend to be used')
	parser.add_argument('--circs_per_job', type=int, default=1, help='provide an integer (default: 5)')
	parser.add_argument('--optim', type=int, default=0, help='provide an integer (default: 5)')
	parser.add_argument('--jobs', type=int, default=5, help='provide an integer (default: 5)')
	parser.add_argument('-t', action="store_true", default=False, help = 'Transpile only')
	args = parser.parse_args()

	IBMQ.load_accounts()
	curdir = os.getcwd()
	input_path = os.path.join(curdir, args.indir)
	output_path = os.path.join(curdir, args.outdir)
	if args.backend == 'statevector_simulator':
		backend = Aer.get_backend('statevector_simulator')
	elif args.backend == 'simulator':
		backend = Aer.get_backend('qasm_simulator')
	else:
		backend = IBMQ.backends(filters = lambda x: x.name() == args.backend).pop()
	print('BACKEND: ', backend.name() )
	simulator = SimManager(input_path, output_path, backend, args.circs_per_job, args.optim, args.jobs)
	
	if args.t == True:
		simulator.transpile()
	else:
		simulator.start()










	

