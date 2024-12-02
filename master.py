import socket
import numpy as np
import threading
import datetime
import time
import os
import pickle
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


class MasterSocket:
    def __init__(self, 
        local_IP: str, 
        local_port: int, 
        slave_port: int, 
        buffer: int, 
        data_node: list
    ):
        self.local_IP = local_IP
        self.local_port = local_port

        self.slave_port = slave_port

        self.buffer = buffer

        self.all_node_opt = ['upload', 'delete', 'modify']
        self.single_node_opt = ['find', 'hash', 'topK']

        self.data_node = data_node
        self.alive_node = None


    def update_alive_dict(self, logger: Logger) -> dict:
        try:
            new_dict = {}
            logger.log(f"[{len(self.data_node)}] slave data node(s) in distribution.")
            for node in self.data_node:
                new_dict[node] = [False, 0]
            logger.log(f"Init all nodes in [{self.data_node}] to dead.")
            return new_dict

        except Exception as e:
            logger.log(e)
            new_dict = {}
            return new_dict


    def run(self, logger: Logger) -> bool:
        try:
            self.alive_node = self.update_alive_dict(logger)

            master_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            master_socket.bind((self.local_IP, self.local_port))
            master_socket.listen(10)
            logger.log(f"Master Socket listens on [{self.local_IP}]-[{self.local_port}], max for 10 incomes.")
        
            try:
                while True:
                    conn, addr = master_socket.accept()
                    
                    if addr[0] in self.data_node:
                        thread = threading.Thread(target=self.heart_beat, 
                            args=(conn, addr, logger))
                        thread.start()
                        logger.log(f"Master connect to Slave [{addr[0]}]-[{addr[1]}].")
                    
                    else:
                        thread = threading.Thread(target=self.both_socket, 
                            args=(conn, addr, logger))
                        thread.start()
                        logger.log(f"Master connect to Client [{addr[0]}]-[{addr[1]}].")

                return True

            except Exception as e:
                logger.log(e)
                master_socket.close()
                return False

        except Exception as e:
            logger.log(e)
            return False


    def both_socket(self, client_conn: socket.socket, client_addr: list, logger: Logger) -> bool:
        try:
            data_from_client = client_conn.recv(self.buffer)
            message_from_client = pickle.loads(data_from_client)
            logger.log(f"Master receive [{message_from_client}] from Client [{client_addr[0]}]-[{client_addr[1]}].")

            _, msg_len = message_from_client.split('#')
            
            data_from_client = client_conn.recv(int(msg_len))
            message_from_client = pickle.loads(data_from_client)
            logger.log(f"Master receive [{message_from_client}] from Client [{client_addr[0]}]-[{client_addr[1]}].")
            msg_to_slave = message_from_client

            if isinstance(msg_to_slave, tuple):
                opt = msg_to_slave[0].split('#')[1]
                print(opt)
            else:
                opt = msg_to_slave.split('#')[1]
                print(opt)

            status_send_back = False
            msg_send_back = 'Empty infomation.'

            if opt in self.all_node_opt:
                all_status = True

                for node in self.data_node:
                    is_alive, _ = self.alive_node[node]
                    if not is_alive:
                        logger.log(f"Slave [{node}] is dead, ignore it.")
                        continue

                    slave_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    try:
                        slave_conn.connect((node, self.slave_port))
                        logger.log(f"Master connect to Slave [{node}]-[{self.slave_port}].")

                        msg_len = len(pickle.dumps(msg_to_slave))
                        msg = f"len#{msg_len}"
                        slave_conn.sendall(pickle.dumps(msg))
                        logger.log(f"Master send [{msg}] to Slave [{node}]-[{self.slave_port}].")
                        time.sleep(0.1)

                        slave_conn.sendall(pickle.dumps(msg_to_slave))
                        logger.log(f"Master send [{msg_to_slave}] to Slave [{node}]-[{self.slave_port}].")

                        data_from_slave = slave_conn.recv(self.buffer)
                        message_from_client = pickle.loads(data_from_slave)
                        logger.log(f"Master receive [{message_from_client}] from Slave [{node}]-[{self.slave_port}].")

                        _, msg_len = message_from_client.split('#')
                        data_from_slave = slave_conn.recv(int(msg_len))
                        message_from_client = pickle.loads(data_from_slave)
                        logger.log(f"Master receive [{message_from_client}] from Slave [{node}]-[{self.slave_port}].")

                        status = message_from_client.split('#')[0]
                        slave_conn.close()

                        if bool(status):
                            continue
                        else:
                            all_status = False
                            break

                    except Exception as e:
                        logger.log(e)
                        status_send_back = False
                        msg_send_back = e

                if all_status:
                    logger.log("Execute Client command successfully.")
                    status_send_back = True
                else:
                    logger.log("Fail to execute Client command.")
                    status_send_back = False
                msg_send_back = message_from_client
            
            elif opt in self.single_node_opt:
                not_done = True
                return_from_slave = None

                for node in self.data_node:
                    is_alive, _ = self.alive_node[node]
                    if not is_alive:
                        logger.log(f"Slave [{node}] is dead, ignore it.")
                        continue
                    
                    slave_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    try:
                        slave_conn.connect((node, self.slave_port))
                        logger.log(f"Master connect to Slave [{node}]-[{self.slave_port}].")

                        msg_len = len(pickle.dumps(msg_to_slave))
                        msg = f"len#{msg_len}"
                        slave_conn.sendall(pickle.dumps(msg))
                        logger.log(f"Master send [{msg}] to Slave [{node}]-[{self.slave_port}].")
                        time.sleep(0.1)

                        slave_conn.sendall(pickle.dumps(msg_to_slave))
                        logger.log(f"Master send [{msg_to_slave}] to Slave [{node}]-[{self.slave_port}].")

                        data_from_slave = slave_conn.recv(self.buffer)
                        message_from_client = pickle.loads(data_from_slave)
                        logger.log(f"Master receive [{message_from_client}] from Slave [{node}]-[{self.slave_port}].")

                        _, msg_len = message_from_client.split('#')
                        data_from_slave = slave_conn.recv(int(msg_len))
                        message_from_slave = pickle.loads(data_from_slave)
                        logger.log(f"Master receive [{message_from_slave}] from Slave [{node}]-[{self.slave_port}].")
                        
                        if opt == 'find':
                            status = message_from_slave[0].split('#')[0]
                        else:
                            status = message_from_slave.split('#')[0]
                        
                        slave_conn.close()

                        if bool(status):
                            not_done = False
                            return_from_slave = message_from_slave
                            break
                        else:
                            continue

                    except Exception as e:
                        logger.log(e)
                        status_send_back = False
                        msg_send_back = str(e)

                if not not_done:
                    logger.log("Fail to execute Client command.")
                    status_send_back = False
                else:
                    logger.log("Execute Client command successfully.")
                    status_send_back = True
                msg_send_back = return_from_slave

            msg_len = len(pickle.dumps(msg_send_back))

            len_msg = f"len#{msg_len}"
            client_conn.sendall(pickle.dumps(len_msg))
            logger.log(f"Master send [{len_msg}] to Client [{client_addr[0]}]-[{client_addr[1]}].")
            time.sleep(0.1)

            client_conn.sendall(pickle.dumps(msg_send_back))
            logger.log(f"Master send [{msg_send_back}] to Client [{client_addr[0]}]-[{client_addr[1]}].")
            client_conn.close()

            return True

        except Exception as e:
            logger.log(e)
            return False


    def heart_beat(self, slave_conn: socket.socket, slave_addr: list, logger: Logger) -> bool:
        try:
            data_from_slave = slave_conn.recv(self.buffer)
            message_from_slave = pickle.loads(data_from_slave)
            logger.log(f"Master receive [{message_from_slave}] from Slave [{slave_addr[0]}]-[{slave_addr[1]}].")

            slave_alive_time = int(message_from_slave)
            self.alive_node[slave_addr[0]] = [True, slave_alive_time]
            logger.log(f"Slave [{slave_addr[0]}] is alive at time [{slave_alive_time}]")

            return True

        except Exception as e:
            logger.log(e)
            return False


    def update_alive(self, sleep_time: int, logger: Logger) -> bool:
        try:
            time.sleep(1)
            while True:
                now_time = int(time.time())
                for node in self.data_node:
                    _, last_alive_time = self.alive_node[node]
                    if now_time - last_alive_time <= sleep_time:
                        logger.log(f"Slave [{node}] is alive.")
                    else:
                        self.alive_node[node] = [False, last_alive_time]
                        logger.log(f"Slave [{node}] is dead at time [{now_time}].")
                logger.log(f"Process update alive sleep for [{sleep_time + 5}] seconds.")
                time.sleep(sleep_time + 5)
            return True
        except Exception as e:
            logger.log(e)
            return False


class Master:
    def __init__(self, 
        master_ip: str, 
        master_port: int, 
        slave_ip: list, 
        slave_port: int
    ):
        self.master_ip = master_ip
        self.master_port = master_port
        self.slave_ip = slave_ip
        self.slave_port = slave_port

        self.logger = Logger('master')

        self.mastersocket = MasterSocket(
            local_IP=self.master_ip, 
            local_port=self.master_port, 
            slave_port=self.slave_port, 
            buffer=1024, 
            data_node=self.slave_ip
        )
        self.logger.log("Create MasterSocket successfully.")


    def threads(self, ) -> bool:
        try:
            ms_run = threading.Thread(target=self.mastersocket.run, args=(self.logger,))
            ms_al = threading.Thread(target=self.mastersocket.update_alive, args=(20, self.logger,))

            ms_run.daemon = True
            ms_al.daemon = True

            ms_run.start()
            ms_al.start()

            self.logger.log(f"Start MasterSocket.run thread [{ms_run}] successfully.")
            self.logger.log(f"Start MasterSocket.update_alive thread [{ms_al}] successfully.")

            while True:
                time.sleep(20)

            return True
        except Exception as e:
            self.logger.log(e)
            return False


if __name__ == '__main__':
    master_ip = '0.0.0.0'  # '10.211.55.74'
    master_port = 65530
    slave_ip = ['slave_IP1', 'slave_IP2', '...']
    slave_port = 65531
    master = Master(master_ip, master_port, slave_ip, slave_port)
    master.threads()
