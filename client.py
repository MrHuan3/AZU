import socket
import numpy as np
import datetime
import time
import pickle
import hashlib
import os
import shutil


class Logger:
	def __init__(self, name: str):
		self.log_file = f'log/{name}_new.log'
		now_time = self.get_time()

		if not os.path.exists('log'):
			os.mkdir('log')
		if not os.path.exists(self.log_file):
			with open('log/last_open_time.log', 'w') as file:
				file.write(now_time)
		else:
			with open('log/last_open_time.log', 'r') as file:
				last_open_time = file.readline()
				shutil.move(self.log_file, f'log/{name}_{last_open_time}.log')
			with open('log/last_open_time.log', 'w') as file:
				file.write(now_time)
		with open(self.log_file, 'w') as file:
			msg = f"[{now_time}] {name} starts\n"
			file.write(msg)
			print(msg)


	def get_time(self) -> str:
		now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		return str(now_time)

	def log(self, msg) -> bool:
		now_time = self.get_time()
		if not isinstance(msg, str):
			msg = str(msg)

		try:
			with open(self.log_file, 'a') as file:
				new_msg = f"[{now_time}] {msg}\n"
				file.write(new_msg)
				# print(new_msg)  # <== For client, print is down, for master and slave, print is on.
				return True
		except Exception as e:
			new_msg = f"[{now_time}] {msg} {e}\n"
			print(new_msg)
			return False


class ClientSocket:
	def __init__(self, master_ip: str, master_port: int, buffer: int):
		self.master_ip = master_ip
		self.master_port = master_port
		self.buffer = buffer


	def run(self, msg_bytes: bytes, logger: Logger) -> tuple[bool, str]:
		try:
			client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			client_socket.connect((self.master_ip, self.master_port))
			logger.log(f"Client connected to Master [{self.master_ip}]-[{self.master_port}].")

			msg_len = len(msg_bytes)
			len_to_master = f"len#{msg_len}"
			client_socket.send(pickle.dumps(len_to_master))
			logger.log(f"Client sent [{len_to_master}] to Master [{self.master_ip}]-[{self.master_port}].")
			time.sleep(0.1)

			client_socket.sendall(msg_bytes)
			logger.log(f"Client sent [{pickle.loads(msg_bytes)}] to Master [{self.master_ip}]-[{self.master_port}].")

			data_from_master = client_socket.recv(self.buffer)
			message_from_master = pickle.loads(data_from_master)
			logger.log(message_from_master)
			msg_len = int(message_from_master.split('#')[1])

			data_from_master = client_socket.recv(msg_len)
			message_from_master = pickle.loads(data_from_master)
			logger.log(message_from_master)

			client_socket.close()

			return True, message_from_master

		except Exception as e:
			logger.log(e)
			return False, str(e)


class Client:
	def __init__(self, name: str, master_ip: str, master_port: int):
		self.name = name
		self.master_ip = master_ip
		self.master_port = master_port

		self.logger = Logger('client')

		self.clientsocket = ClientSocket(self.master_ip, self.master_port, 1024)
		self.logger.log("Create ClientSocket successfully.")


	def threads(self, ) -> bool:
		try:
			self.logger.log(f"Start Client thread successfully.")
			self.logger.log(f"Start ClientSocket thread [{self.clientsocket}] successfully.")

			self.run()

			return True
		except Exception as e:
			self.logger.log(e)
			return False


	def upload_vector(self) -> tuple[bool, str, np.array, str]:
		try:
			key = input("Enter unique key for input vector: ")
			key = str(key)
			self.logger.log(f"[{self.name}] enter key [{key}] for input vector.")

			input_path = input("Enter new local vector path: ")
			input_path = str(input_path)

			if input_path[0] == '/':
				if os.path.exists(input_path):
					if input_path[-4:] == '.npy':
						vector = np.load(input_path)
						full_path = input_path
					else:
						return False, '', None, f"Expect to input .npy file, but got [.{input_path.split('.')[-1]}] file."
				else:
					return False, '', None, f"[{input_path}] is invalid."
			else:
				pwd = os.getcwd()
				abs_target_vector_path = os.path.join(pwd, input_path)
				if os.path.exists(abs_target_vector_path):
					if abs_target_vector_path[-4:] == '.npy':
						vector = np.load(abs_target_vector_path)
						full_path = abs_target_vector_path
					else:
						return False, '', None, f"Expect to input .npy file, but got [.{abs_target_vector_path.split('.')[-1]}] file."
				else:
					return False, '', None, f"[{abs_target_vector_path}] is invalid."

			return True, key, vector, f"[{self.name}] upload vector [{full_path}] with key [{key}]."

		except Exception as e:
			return False, '', None, str(e)


	def delete_vector(self) -> tuple[bool, str, str]:
		try:
			target_key = input("Enter key of target vector: ")
			target_key = str(target_key)

			return True, target_key, f"[{self.name}] enter target vector [{target_key}]."

		except Exception as e:
			return False, '', str(e)


	def modify_vector(self) -> tuple[bool, str, np.array, str, str]:
		try:
			target_key = input("Enter key of target vector: ")
			target_key = str(target_key)
			self.logger.log(f"[{self.name}] enter target vector [{target_key}].")
			input_path = input("Enter new local vector path: ")
			input_path = str(input_path)

			if input_path[0] == '/':
				if os.path.exists(input_path):
					if input_path[-4:] == '.npy':
						vector = np.load(input_path)
					else:
						return False, '', None, '', f"Expect to input .npy file, but got [.{input_path.split('.')[-1]}] file."
				else:
					return False, '', None, '', f"[{input_path}] is invalid."
			else:
				pwd = os.getcwd()
				abs_target_vector_path = os.path.join(pwd, input_path)
				if os.path.exists(abs_target_vector_path):
					if abs_target_vector_path[-4:] == '.npy':
						vector = np.load(abs_target_vector_path)
					else:
						return False, '', None, '', f"Expect to input .npy file, but got [.{abs_target_vector_path.split('.')[-1]}] file."
				else:
					return False, '', None, '', f"[{abs_target_vector_path}] is invalid."

			print("Enter command, where INPUT_VECTOR stands for input vector, TARGET_VECTOR stands for target vector.")
			cmd = input("(Example: TARGET_VECTOR[0:2,3:6] += INPUT_VECTOR): ")
			cmd = str(cmd)
			if 'TARGET_VECTOR' not in cmd or 'INPUT_VECTOR' not in cmd:
				return False, '', None, '', f"No 'TARGET_VECTOR' or 'INPUT_VECTOR' is found in [{cmd}]."

			return True, target_key, vector, cmd, f"[{self.name}] enter command [{cmd}]. With target vector key [{target_key}]"

		except Exception as e:
			return False, '', None, '', str(e)


	def find_vector(self) -> tuple[bool, str, str]:
		try:
			target_key = input("Enter key of target vector: ")
			target_key = str(target_key)
			return True, target_key, f"[{self.name}] input key [{target_key}]."

		except Exception as e:
			return False, '', str(e)


	def cal_vector_hash(self, vector: np.array) -> tuple[bool, str, str]:
		try:
			vector_bytes = vector.flatten().tobytes()
			hash_value = hashlib.sha256(vector_bytes).hexdigest()

			return True, hash_value, f"Data maps to sha256 value [{hash_value}]."

		except Exception as e:
			return False, '', str(e)


	def find_hash_fn(self, path: str) -> tuple[bool, str, str]:
		try:
			if os.path.exists(path):
				if path[-4:] == '.npy':
					vector = np.load(path)
					OK_cal_vector_hash, hash_value, msg = self.cal_vector_hash(vector)
					return OK_cal_vector_hash, hash_value, msg
				else:
					return False, '', f"Expect to input .npy file, but got [.{path.split('.')[-1]}] file."
			else:
				return False, '', f"[{path}] is invalid."

		except Exception as e:
			return False, '', str(e)


	def find_hash(self) -> tuple[bool, str, str]:
		try:
			target_vector_path = input("Enter target vector path: ")
			if target_vector_path[0] == '/':
				OK_find_hash_fn, vector, msg = self.find_hash_fn(target_vector_path)
				return OK_find_hash_fn, vector, msg
			else:
				pwd = os.getcwd()
				abs_target_vector_path = os.path.join(pwd, target_vector_path)
				OK_find_hash_fn, vector, msg = self.find_hash_fn(abs_target_vector_path)
				return OK_find_hash_fn, vector, msg

		except Exception as e:
			return False, '', str(e)


	def topK_fn(self, vector_path: str) -> tuple[bool, np.array, str]:
		try:
			if os.path.exists(vector_path):
				if vector_path[-4:] == '.npy':
					K = input("Enter K: ")
					try:
						K = int(K)
					except Exception as e:
						self.logger.log(e)
						return False, None, ''

					if K > 0:
						vector = np.load(vector_path)
						self.logger.log(f"[{self.name}] amin to find top-[{K}] vectors similar to [{vector_path}].")
						return True, vector, str(K)
					else:
						return False, None, f"Expect to int a positive int, but [{K}] is not positive."
				else:
					return False, None, f"Expect to input .npy file, but got [.{vector_path.split('.')[-1]}] file."
			else:
				return False, None, f"[{vector_path}] is invalid."

		except Exception as e:
			return False, None, str(e)


	def topK(self) -> tuple[bool, np.array, str]:
		try:
			target_vector_path = input("Enter target vector path: ")
			if target_vector_path[0] == '/':
				OK_topK_fn, vector, msg = self.topK_fn(target_vector_path)
				return OK_topK_fn, vector, msg
			else:
				pwd = os.getcwd()
				abs_target_vector_path = os.path.join(pwd, target_vector_path)
				OK_topK_fn, vector, msg = self.topK_fn(abs_target_vector_path)
				return OK_topK_fn, vector, msg
				
		except Exception as e:
			return False, None, str(e)


	def help(self) -> bool:
		try:
			self.logger.log("Show help list.")
			print("(1) Upload numpy vector.")
			print("(2) Delete target vector.")
			print("(3) Modify target vector.")
			print("(4) Look up target vector.")
			print("(5) Find target hash vector.")
			print("(6) Find top-K similar vectors.")
			print("(7) Help")
			print("(8) Quit.")
			return True
		except Exception as e:
			self.logger.log(e)
			return False


	def quit(self) -> tuple[bool, str]:
		try:
			return True, "Bye~"

		except Exception as e:
			return False, str(e)


	def get_input(self) -> tuple[bool, bytes]:
		try:
			user_input = input("Choose operation, enter number: ")
			try:
				user_input = int(user_input)
			except Exception as e:
				self.logger.log(e)
				msg = f"Expect to input int, but got [{user_input}]"
				return False, pickle.dumps(msg)

			if 1 <= user_input <= 8:
				if user_input == 1:
					self.logger.log(f"[{self.name}] input [1] to upload new vector.")
					signal, key, vector, msg = self.upload_vector()
					self.logger.log(msg)

					cmd_vector = (f"{self.name}#upload#{key}", vector)
					cmd_vector_stream = pickle.dumps(cmd_vector)
					user_cmd = cmd_vector_stream

				elif user_input == 2:
					self.logger.log(f"[{self.name}] input [2] to delete target vector.")
					signal, key, msg = self.delete_vector()
					self.logger.log(msg)

					user_cmd = pickle.dumps(f"{self.name}#delete#{key}")

				elif user_input == 3:
					self.logger.log(f"[{self.name}] input [3] to modify target vector.")
					signal, key, new_vector, cmd, msg = self.modify_vector()
					self.logger.log(msg)

					cmd_vector = (f"{self.name}#modify#{key}#{cmd}", new_vector)
					cmd_vector_stream = pickle.dumps(cmd_vector)
					user_cmd = cmd_vector_stream

				elif user_input == 4:
					self.logger.log(f"[{self.name}] input [4] to find target vector.")
					signal, key, msg = self.find_vector()
					self.logger.log(msg)

					user_cmd = pickle.dumps(f"{self.name}#find#{key}")

				elif user_input == 5:
					self.logger.log(f"[{self.name}] input [5] to find hash vector.")
					signal, hash_value, msg = self.find_hash()
					self.logger.log(msg)

					user_cmd = pickle.dumps(f"{self.name}#hash#{hash_value}")

				elif user_input == 6:
					self.logger.log(f"[{self.name}] input [6] to find top-K.")
					signal, vector, msg = self.topK()
					self.logger.log(msg)

					cmd_vector = (f"{self.name}#topK#{msg}", vector)
					cmd_vector_stream = pickle.dumps(cmd_vector)
					user_cmd = cmd_vector_stream

				elif user_input == 7:
					self.logger.log(f"[{self.name}] input [7] for help.")
					signal = self.help()
					return signal, pickle.dumps('help')

				elif user_input == 8:
					self.logger.log(f"[{self.name}] input [8] to quit.")
					signal, msg = self.quit()
					self.logger.log(msg)
					return signal, pickle.dumps('quit')

				return signal, user_cmd

			else:
				msg = f"No operation mapping to input [{user_input}] was selected."
				self.logger.log(msg)
				print(msg)
				return False, pickle.dumps(msg)

		except Exception as e:
			self.logger.log(e)
			return False, pickle.dumps(str(e))


	def run(self) -> bool:
		try:
			print("WELCOME TO AZU")
			self.help()
			while True:
				OK_get_input, msg_bytes = self.get_input()

				if OK_get_input and pickle.loads(msg_bytes) == 'quit':
					return True
				elif OK_get_input and pickle.loads(msg_bytes) == 'help':
					continue
				elif OK_get_input:
					OK_run, msg_from_master = self.clientsocket.run(msg_bytes, self.logger)
					print(msg_from_master)
					if isinstance(msg_from_master, tuple):
						vector = msg_from_master[1]
						key = pickle.loads(msg_bytes).split('#')[2]
						if not os.path.exists('download'):
							os.mkdir('download')
							self.logger.log("make directory 'download' successful.")
						save_path = os.path.join('download', f"{key}.npy")
						np.save(save_path, vector)
						msg = f"Save download vector [{key}] to file [{save_path}]."
						self.logger.log(msg)
						print(msg)
				else:
					print(pickle.loads(msg_bytes))

			return True
		except Exception as e:
			print(e)
			self.logger.log(e)
			return False



if __name__ == '__main__':
	name = 'guest'  # ['admin', 'guest', 'other']
	master_ip = '10.211.55.74'
	master_port = 65530
	client = Client(name, master_ip, master_port)
	client.threads()

