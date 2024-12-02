import socket
import numpy as np
import threading
import datetime
import time
import hashlib
import pickle
import faiss
import os
import glob
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
                print(new_msg)
                return True
        except Exception as e:
            new_msg = f"[{now_time}] {msg} {e}\n"
            print(new_msg)
            return False


class Data:
    def __init__(self, key: str,
        create_time: str,
        owner: str,
        path: str,
    ):

        self.key = key  # unique key value
        self.create_time = create_time  # data create time
        self.last_modified_time = create_time  # last modified time
        self.owner = owner  # which owner this data belongs to
        self.vector_data = None  # vector data
        self.data_shape = None  # vector shape
        self.size = None  # disk space this data takes
        self.path = path  # path to the data locally
        self.key_hash = None  # hash of key
        self.data_hash = None  # hash of data


    def update_key_hash(self, hash_value: str) -> tuple[bool, str]:
        try:
            self.key_hash = hash_value
            return True, f"Update key hash of [{self.key}] successfully."
        except Exception as e:
            return False, str(e)


    def update_data_hash(self, hash_value: str) -> tuple[bool, str]:
        try:
            self.data_hash = hash_value
            return True, f"Update data hash of [{self.key}] successfully."
        except Exception as e:
            return False, str(e)


    def save_data(self, vector: np.array) -> tuple[bool, str]:
        try:
            np.save(self.path, vector)
            return True, f"Save [{self.key}] data to [{self.path}] successfully."
        except Exception as e:
            return False, str(e)


    def get_shape(self, vector: np.array) -> tuple[bool, str]:
        try:
            self.data_shape = vector.shape
            return True, f"The shape of [{self.key}] is [{self.data_shape}]."
        except Exception as e:
            return False, str(e)


    def get_size(self) -> tuple[bool, str]:
        try:
            file_size = os.path.getsize(self.path)

            KB = 1024
            MB = KB * 1024
            GB = MB * 1024
            TB = GB * 1024

            if file_size < KB:
                self.size = f"{file_size} bytes"
            elif file_size < MB:
                self.size = f"{file_size / KB:.2f} KB"
            elif file_size < GB:
                self.size = f"{file_size / MB:.2f} KB"
            elif file_size < TB:
                self.size = f"{file_size / GB:.2f} GB"
            else:
                self.size = f"{file_size / TB:.2f} TB"
            return True, f"Space [{self.key}] takes on the disk is [{self.size}]."

        except Exception as e:
            return False, str(e)


    def modify_data(self, cmd: str, new_data: np.array) -> tuple[bool, np.array, str]:
        try:
            ori_data = np.load(self.path)
            exec(cmd, {'TARGET_VECTOR': ori_data, 'INPUT_VECTOR': new_data})
            np.save(self.path, ori_data)

            return True, ori_data, f"Execute [{cmd}] successfully."

        except Exception as e:
            return False, None, str(e)


    def look_data(self) -> tuple[bool, np.array, str]:
        try:
            look_data = np.load(self.path)

            return True, look_data, f"Look [{self.key}] data locally."

        except Exception as e:
            return False, None, str(e)


    def update_last_modified_time(self, time: str) -> tuple[bool, str]:
        try:
            self.last_modified_time = time
            return True, f"Update [{self.key}] last modified time to [{time}]."
        except Exception as e:
            return False, str(e)


class Handler:
    def __init__(self):
        self.keys = {}
        self.data_hashes = {}


    def rebuild(self, logger: Logger) -> bool:
        try:
            rebuild_num = 0
            index_root = 'index/'
            if not os.path.exists(index_root):
                os.mkdir(index_root)
            for ind in glob.glob(index_root + '/*.pkl'):
                with open(ind, 'rb') as file:
                    index = pickle.load(file)
                    key, data_path = index.key, index.path
                    index_path = data_path.replace('data/', 'index/').replace('.npy', '.pkl')
                    hash_value = data_path.split('/')[-1].split('.')[0]
                    self.keys[key] = [hash_value, index_path, data_path]
                    self.data_hashes[hash_value] = [key, index_path, data_path]
                    rebuild_num += 1
            logger.log(f"Rebuild [{rebuild_num}] data from disk successfully.")
            return True
        except Exception as e:
            logger.log(e)
            return False


    def handle(self, msg, logger: Logger) -> tuple[bool, bytes]:
        try:
            if isinstance(msg, str):
                opt = msg.split('#')[1]
            elif isinstance(msg, tuple):
                cmd, vector = msg[0], msg[1]
                opt = cmd.split('#')[1]

            print(opt, msg)

            if opt == 'upload':
                signal, return_msg_bytes = self.create_data(cmd, vector, logger)

            elif opt == 'delete':
                signal, return_msg_bytes = self.delete_data(msg, logger)

            elif opt == 'modify':
                signal, return_msg_bytes = self.modify_data(cmd, vector, logger)

            elif opt == 'find':
                signal, return_msg_bytes = self.look_data(msg, logger)

            elif opt == 'hash':
                signal, return_msg_bytes = self.find_hash_data(msg, logger)

            elif opt == 'topK':
                signal, return_msg_bytes = self.find_similar_topK(cmd, vector, logger)

            return signal, return_msg_bytes

        except Exception as e:
            logger.log(e)
            return False, pickle.dumps(str(e))


    def get_time(self) -> str:
        now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return str(now_time)


    def create_data(self, cmd: str, vector: np.array, logger: Logger) -> tuple[bool, bytes]:
        try:
            name, opt, key = cmd.split('#')
            if name != 'admin':
                return_msg = f"False#[{name}] have no privilege to upload data. This operation is logged."
                logger.log(return_msg)
                return False, pickle.dumps(return_msg)
            if key in self.keys.keys():
                msg = f"[{key}] is not unique. Fail to create data."
                logger.log(msg)
                return_msg = f"{False}#{msg}"
                return False, pickle.dumps(return_msg)
            else:
                now_time = self.get_time()
                OK_cal_key_hash, key_hash_value, msg = self.cal_key_hash(key)
                logger.log(msg)
                tensor_save_path = os.path.join('data', f"{key_hash_value}.npy")

                new_data = Data(key,
                    now_time,
                    name,
                    tensor_save_path
                )

                logger.log(f"Create data [{key}] successfully.")

                OK_cal_vector_hash, data_hash_value, msg = self.cal_vector_hash(vector)
                if not OK_cal_vector_hash:
                    logger.log(msg)
                else:
                    OK_update_data_hash, msg = new_data.update_data_hash(data_hash_value)
                    logger.log(msg)

                OK_save_data, msg = new_data.save_data(vector)
                logger.log(msg)

                OK_get_shape, msg = new_data.get_shape(vector)
                logger.log(msg)

                OK_get_size, msg = new_data.get_size()
                logger.log(msg)

                OK_save_index, msg = self.save_index(new_data)
                logger.log(msg)

                return_msg = f"{OK_save_index}#{msg}"
                return True, pickle.dumps(return_msg)

        except Exception as e:
            logger.log(e)
            return_msg = f"{False}#{e}"
            return False, pickle.dumps(return_msg)


    def cal_key_hash(self, key: str) -> tuple[bool, str, str]:
        try:
            key_value = key.encode('utf-8')
            hash_value = hashlib.sha256(key_value).hexdigest()

            return True, hash_value, f"[{key}] maps to sha256 value [{hash_value}]."

        except Exception as e:
            return False, '', str(e)


    def cal_vector_hash(self, vector: np.array) -> tuple[bool, str, str]:
        try:
            vector_bytes = vector.flatten().tobytes()
            hash_value = hashlib.sha256(vector_bytes).hexdigest()

            return True, hash_value, f"Data maps to sha256 value [{hash_value}]."

        except Exception as e:
            return False, '', str(e)


    def save_index(self, index: Data) -> tuple[bool, str]:
        try:
            index_path = index.path.replace('data/', 'index/').replace('.npy', '.pkl')

            with open(index_path, 'wb') as file:
                pickle.dump(index, file)
            data_hash = index.data_hash
            self.keys[index.key] = [data_hash, index_path, index.path]
            self.data_hashes[data_hash] = [index.key, index_path, index.path]

            return True, f"Save index [{index.key}] to [{index.path}] successfully."

        except Exception as e:
            return False, str(e)


    def delete_data(self, cmd: str, logger: Logger) -> tuple[bool, bytes]:
        try:
            name, opt, key = cmd.split('#')
            if name != 'admin':
                msg = f"[{name}] has no privilege to delete data. This operation is logged."
                logger.log(msg)
                return_msg = f"{False}#{msg}"
                return False, pickle.dumps(return_msg)

            if key in self.keys.keys():
                index_path = self.keys[key][1]

                with open(index_path, 'rb') as file:
                    index = pickle.load(file)
                data_owner = index.owner

                if data_owner != name:
                    msg = f"[{name}] has no privilege to delete other's data. This operation is logged."
                    logger.log(msg)
                    return_msg = f"{False}#{msg}"
                    return False, pickle.dumps(return_msg)

                else:
                    OK_delete_data, msg = self.delete_target_data(key)
                    logger.log(msg)
                    return_msg = f"{OK_delete_data}#{msg}"
                    return OK_delete_data, pickle.dumps(return_msg)
            else:
                msg = f"[{key}] is not stored here."
                logger.log(msg)
                return_msg = f"{False}#{msg}"
                return False, pickle.dumps(return_msg)

        except Exception as e:
            logger.log(e)
            return_msg = f"{False}#{e}"
            return False, pickle.dumps(return_msg)


    def delete_target_data(self, key: str) -> tuple[bool, str]:
        try:
            data_hash_value, index_path, data_path = self.keys[key]
            del self.data_hashes[data_hash_value]
            os.remove(index_path)
            os.remove(data_path)
            del self.keys[key]
            return True, f"Data [{key}-{data_hash_value}] has been deleted successfully."
        except Exception as e:
            return False, str(e)


    def modify_data(self, input_msg: str, vector: np.array, logger: Logger) -> tuple[bool, bytes]:
        try:
            name, opt, key, cmd = input_msg.split('#')
            if name != 'admin':
                msg = f"[{name}] has no privilege to modify data. This operation is logged."
                logger.log(msg)
                return_msg = f"{False}#{msg}"
                return False, pickle.dumps(return_msg)

            if key in self.keys.keys():
                index_path = self.keys[key][1]

                with open(index_path, 'rb') as file:
                    index = pickle.load(file)
                data_owner = index.owner

                if data_owner != name:
                    msg = f"[{name}] has no privilege to modify other's data. This operation is logged."
                    logger.log(msg)
                    return_msg = f"{False}#{msg}"
                    return False, pickle.dumps(return_msg)

                else:
                    OK_modify_data, msg = self.modify_target_data(key, cmd, vector, logger)
                    logger.log(msg)
                    return_msg = f"{OK_modify_data}#{msg}"
                    return False, pickle.dumps(return_msg)
            else:
                msg = f"[{key}] is not stored here."
                logger.log(msg)
                return_msg = f"{False}#{msg}"
                return False, pickle.dumps(return_msg)

        except Exception as e:
            logger.log(e)
            return_msg = f"{False}#{e}"
            return False, pickle.dumps(return_msg)


    def modify_target_data(self, key: str, cmd: str, new_data: np.array, logger: Logger) -> tuple[bool, str]:
        try:
            _, index_path, data_path = self.keys[key]
            with open(index_path, 'rb') as file:
                index = pickle.load(file)

            OK_modify_data, modified_data, msg = index.modify_data(cmd, new_data)
            logger.log(msg)

            if OK_modify_data:
                OK_cal_vector_hash, data_hash_value, msg = self.cal_vector_hash(modified_data)
                logger.log(msg)
                OK_update_data_hash, msg = index.update_data_hash(data_hash_value)
                logger.log(msg)

                now_time = self.get_time()
                OK_update_last_modified_time, msg = index.update_last_modified_time(now_time)
                logger.log(msg)

                with open(index_path, 'wb') as file:
                    pickle.dump(index, file)

                return OK_modify_data, f"Modify [{key}] done."

            else:
                return OK_modify_data, msg
        except Exception as e:
            return False, str(e)


    def look_data(self, cmd: str, logger: Logger) -> tuple[bool, bytes]:
        try:
            name, opt, key = cmd.split('#')
            if name == 'other':
                msg = f"[{name}] has no privilege to look data. This operation is logged."
                logger.log(msg)
                return_msg = f"{False}#{msg}"
                return False, pickle.dumps(return_msg)

            if key in self.keys.keys():
                index_path = self.keys[key][1]

                with open(index_path, 'rb') as file:
                    index = pickle.load(file)
                data_owner = index.owner

                if data_owner != name:
                    msg = f"[{name}] has no privilege to look other's data. This operation is logged."
                    logger.log(msg)
                    return_msg = f"{False}#{msg}"
                    return False, pickle.dumps(return_msg)

                else:
                    OK_look_data, msg, vector = self.look_target_data(key, logger)
                    logger.log(msg)
                    return_msg = (f"{OK_look_data}#{msg}", vector)
                    return OK_look_data, pickle.dumps(return_msg)
            else:
                msg = f"[{key}] is not stored here."
                logger.log(msg)
                return_msg = f"{False}#{msg}"
                return False, pickle.dumps(return_msg)

        except Exception as e:
            logger.log(e)
            return_msg = f"{False}#{e}"
            return False, pickle.dumps(return_msg)


    def look_target_data(self, key: str, logger: Logger) -> tuple[bool, str, np.array]:
        try:
            _, index_path, data_path = self.keys[key]
            with open(index_path, 'rb') as file:
                index = pickle.load(file)

            OK_look_data, look_data, msg = index.look_data()
            logger.log(msg)

            if OK_look_data:
                now_time = self.get_time()
                OK_update_last_modified_time, msg = index.update_last_modified_time(now_time)
                logger.log(msg)

                with open(index_path, 'wb') as file:
                    pickle.dump(index, file)

                return OK_look_data, f"Return [{key}] data.", look_data

            else:
                return OK_look_data, f"Fail to look [{key}] data.", None

        except Exception as e:
            logger.log(e)
            return False, str(e), None


    def find_hash_data(self, cmd: str, logger: Logger) -> tuple[bool, bytes]:
        try:
            name, opt, hash_value = cmd.split('#')
            if name == 'other':
                msg = f"[{name}] has no privilege to find data. This operation is logged."
                logger.log(msg)
                return_msg = f"{False}#{msg}"
                return False, pickle.dumps(return_msg)

            if hash_value in self.data_hashes.keys():
                index_path = self.data_hashes[hash_value][1]

                with open(index_path, 'rb') as file:
                    index = pickle.load(file)
                data_owner = index.owner

                if data_owner !=name:
                    msg = "No target data found."
                    logger.log(msg)
                    return_msg = f"{False}#{msg}"
                    return False, pickle.dumps(return_msg)

                else:
                    OK_find_hash_data, key, msg = self.find_target_hash_data(hash_value, logger)
                    logger.log(msg)
                    return_msg = f"{OK_find_hash_data}#{msg}"
                    return False, pickle.dumps(return_msg)
            else:
                msg = f"[{hash_value}] is not stored here."
                logger.log(msg)
                return_msg = f"{False}#{msg}"
                return False, pickle.dumps(return_msg)

        except Exception as e:
            logger.log(e)
            return_msg = f"{False}#{e}"
            return False, pickle.dumps(return_msg)


    def find_target_hash_data(self, data_hash: str, logger: Logger) -> tuple[bool, str, str]:
        try:
            key, index_path, _ = self.data_hashes[data_hash]
            with open(index_path, 'rb') as file:
                index = pickle.load(file)

            now_time = self.get_time()
            OK_update_last_modified_time, msg = index.update_last_modified_time(now_time)
            logger.log(msg)

            with open(index_path, 'wb') as file:
                pickle.dump(index, file)

            return True, key, f"Find hash [{data_hash}] with key [{key}]"

        except Exception as e:
            return False, '', str(e)


    def find_similar_topK(self, cmd: str, vector: np.array, logger: Logger) -> tuple[bool, bytes]:
        try:
            name, opt, K = cmd.split('#')
            K = int(K)
            if name == 'other':
                msg = f"[{name}] has no privilege to find data. This operation is logged."
                logger.log(msg)
                return_msg = f"{False}#{msg}"
                return False, pickle.dumps(return_msg)

            OK_find_topK, topK_list, msg = self.find_topK(name, vector, K, logger)
            logger.log(msg)

            if OK_find_topK:
                return_msg = f"{OK_find_topK}#{msg}"
                return True, pickle.dumps(return_msg)

            else:
                msg = f"Fail to find top-[{K}] similar data."
                logger.log(msg)
                return_msg = f"{False}#{msg}"
                return False, pickle.dumps(return_msg)

        except Exception as e:
            logger.log(e)
            return_msg = f"{False}#{e}"
            return False, pickle.dumps(return_msg)


    def find_topK(self, name: str, vector: np.array, K: int, logger: Logger) -> tuple[bool, list, str]:
        try:
            target_vector_shape = vector.shape
            key_list = []
            data_list = []
            for key in self.keys.keys():
                data_hash, index_path, data_path = self.keys[key]
                with open(index_path, 'rb') as file:
                    index = pickle.load(file)
                if index.owner == name and index.data_shape == target_vector_shape:
                    data = np.load(data_path)
                    key_list.append(index.key)
                    data_list.append(data.flatten()[None, :])
                print(index.owner, name, index.data_shape, target_vector_shape)

            if len(key_list) < K:
                return True, key_list, f"Only [{len(key_list)}] data found, return them all back {key_list}."
            else:
                flatten_target = vector.flatten()[None, :]
                data_bank = np.concatenate(data_list)
                dimension = flatten_target.shape[-1]
                topK_index = faiss.IndexFlatL2(dimension)
                topK_index.add(data_bank)
                dis, ind = topK_index.search(flatten_target, K)

                select_key = []
                ind = ind.tolist()[0]
                for num in ind:
                    select_key.append(key_list[num])

                return True, select_key, f"Find top-[{K}] similar data with key [{select_key}] and instances [{dis}]."

        except Exception as e:
            logger.log(e)
            return False, [], "Fail to find similar data."


class SlaveSocket:
    def __init__(self, local_ip: str, local_port: int, remote_ip: str, remote_port: int, buffer: int=1024):
        self.local_ip = local_ip
        self.local_port = local_port
        self.remote_ip = remote_ip
        self.remote_port = remote_port
        self.buffer = buffer


    def send(self, logger: Logger) -> bool:
        try:
            retry = 0
            inter = 5
            heart_beats_inter = 20
            while True:
                if retry > 0:
                    logger.log(f"Retry [{retry}] time, send heart beat.")
                try:
                    now_time = int(time.time())
                    slave_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    slave_socket.connect((self.remote_ip, self.remote_port))
                    msg = pickle.dumps(f"{now_time}")
                    slave_socket.sendall(msg)
                    logger.log(f"Slave sent heart-beat to Master [{self.remote_ip}]-[{self.remote_port}] as [{now_time}].")
                    OK_send_heart_beat = True
                    slave_socket.close()
                except Exception as e:
                    logger.log(e)
                    OK_send_heart_beat = False

                if OK_send_heart_beat:
                    retry = 0
                    logger.log("Send heart beat.")
                    time.sleep(heart_beats_inter)
                else:
                    logger.log("Fail to send heart beat. Waiting for retry.")
                    retry += 1
                    time.sleep(int(inter * retry))
            return True

        except Exception as e:
            logger.log(e)
            return False


    def receive(self, handler: Handler, logger: Logger) -> bool:
        try:
            slave_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            slave_socket.bind((self.local_ip, self.local_port))
            slave_socket.listen(10)
            logger.log(f"Slave Socket listens on [{self.local_ip}]-[{self.local_port}], max for 10 incomes.")

            try:
                while True:
                    conn, addr = slave_socket.accept()

                    data_from_master = conn.recv(self.buffer)
                    message_from_master = pickle.loads(data_from_master)
                    logger.log(f"Slave receive [{message_from_master}] from Client [{addr[0]}]-[{addr[1]}].")

                    _, msg_len = message_from_master.split('#')
                    data_from_master = conn.recv(int(msg_len))
                    message_from_master = pickle.loads(data_from_master)
                    logger.log(f"Master receive [{message_from_master}] from Client [{addr[0]}]-[{addr[1]}].")

                    OK_handle, return_bytes = handler.handle(message_from_master, logger)

                    return_len = len(return_bytes)
                    return_len_msg = pickle.dumps(f"len#{return_len}")
                    conn.sendall(return_len_msg)
                    logger.log(f"Slave sent [{pickle.loads(return_len_msg)}] to Master [{addr[0]}]-[{addr[1]}].")
                    time.sleep(0.1)

                    conn.sendall(return_bytes)
                    logger.log(f"Slave sent [{pickle.loads(return_bytes)}] to Master [{addr[0]}]-[{addr[1]}].")

                return True

            except Exception as e:
                logger.log(e)
                slave_socket.close()
                return False

        except Exception as e:
            logger.log(e)
            return False


class Slave:
    def __init__(self, slave_ip: str, slave_port: int, master_ip: str, master_port: int):
        self.slave_ip = slave_ip
        self.slave_port = slave_port
        self.master_ip = master_ip
        self.master_port = master_port

        self.heart_beats_inter = 20

        self.logger = Logger('slave')


    def threads(self, ) -> bool:
        try:
            slavesocket = SlaveSocket(self.slave_ip, self.slave_port, self.master_ip, self.master_port)
            handler = Handler()

            handler.rebuild(self.logger)

            ss_send = threading.Thread(target=slavesocket.send, args=(self.logger, ))
            ss_recv = threading.Thread(target=slavesocket.receive, args=(handler, self.logger, ))

            ss_send.daemon = True
            ss_recv.daemon = True

            ss_send.start()
            ss_recv.start()

            self.logger.log(f"Start SlaveSend.run thread [{ss_send}] successfully.")
            self.logger.log(f"Start SlaveReceive.run thread [{ss_recv}] successfully.")
            
            while True:
                time.sleep(20)

            return True
        except Exception as e:
            self.logger.log(e)
            return False


if __name__ == '__main__':
    slave_ip = '0.0.0.0'  # '10.211.55.75' '10.211.55.76'
    slave_port = 65531
    master_ip = 'master_IP'
    master_port = 65530
    slave = Slave(slave_ip, slave_port, master_ip, master_port)
    slave.threads()

