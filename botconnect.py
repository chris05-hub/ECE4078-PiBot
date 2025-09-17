import socket
import struct
import threading
import time
import numpy as np
import cv2


class BotConnect:
    def __init__(self, robot_ip, wheel_port=8000, camera_port=8001, pid_config_port=8002):
        self.robot_ip = robot_ip
        self.wheel_port = wheel_port
        self.camera_port = camera_port
        self.pid_config_port = pid_config_port
        
        # Connection state
        self.running = True
        
        # Robot state
        self.left_speed = 0
        self.right_speed = 0
        self.left_count = 0
        self.right_count = 0
        
        # Camera frame
        self.frame = None
        self.frame_lock = threading.Lock()
        
        # Start connection threads
        self.command_thread = threading.Thread(target=self._connect_wheel)
        self.command_thread.daemon = True
        self.command_thread.start()
        self.camera_thread = threading.Thread(target=self._connect_camera)
        self.camera_thread.daemon = True
        self.camera_thread.start()
    
    def set_pid(self, use_pid, kp, ki, kd):
        """Send PID constants to the robot"""
        try:
            # Open a temporary socket for PID configuration
            pid_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            pid_socket.connect((self.robot_ip, self.pid_config_port))
            
            # Pack PID constants and send
            pid_data = struct.pack("!ffff", use_pid, kp, ki, kd)
            pid_socket.sendall(pid_data)
            
            # Wait for acknowledgment
            response = pid_socket.recv(4)
            status = struct.unpack("!i", response)[0]
            
            pid_socket.close()
            return status == 1
        except Exception as e:
            print(f"Failed to set PID: {str(e)}")
            return False      
    
    def set_velocity(self, wheel_speed):
        # Change the robot speed here. The value should be between -1 and 1.
        # Note that this is just a number specifying how fast the robot should go, not the actual speed in m/s
        self.left_speed = max(min(wheel_speed[0], 1), -1) 
        self.right_speed = max(min(wheel_speed[1], 1), -1)
        return self.left_speed, self.right_speed
    
    def get_image(self):
        with self.frame_lock: # need to lock when multiple threads access the same data, especially if data is bigger in size
            return self.frame.copy() if self.frame is not None else None
    
    def get_encoder_counts(self):
        return self.left_count, self.right_count
    
    def stop(self):
        self.running = False
        self.set_velocity([0,0])
        time.sleep(0.2)
    
    def _connect_wheel(self):
        # Thread function to handle wheel connection (sending speed and receiving encoder counts)
        while self.running:
            try:
                wheel_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                wheel_socket.connect((self.robot_ip, self.wheel_port))
                print("Connected to wheel server")
                
                while self.running:
                    try:
                        # Send current speed values
                        wheel_speed = struct.pack("!ff", self.left_speed, self.right_speed)
                        wheel_socket.sendall(wheel_speed)
                        
                        # Receive encoder counts
                        data = wheel_socket.recv(8)
                        if not data or len(data) != 8:
                            print("Wheel server disconnected")
                            break
                        
                        # Update encoder counts
                        self.left_count, self.right_count = struct.unpack("!ii", data)
                        
                        # Small delay to avoid flooding the network
                        time.sleep(0.05)
                        
                    except Exception as e:
                        print(f"Wheel error: {str(e)}")
                        break
                    
                wheel_socket.close()
                
            except Exception as e:
                print(f"Wheel connection error: {str(e)}")
                time.sleep(1)  # Wait before trying to reconnect
    
    def _connect_camera(self):
        # Thread function to handle camera connection
        while self.running:
            try:
                camera_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                camera_socket.connect((self.robot_ip, self.camera_port))
                print("Connected to camera server")
                
                while self.running:
                    try:
                        size_data = camera_socket.recv(4, socket.MSG_WAITALL)
                        if not size_data or len(size_data) != 4:
                            print("Camera server disconnected")
                            break
                        
                        # Unpack data size
                        jpeg_size = struct.unpack("!I", size_data)[0]
                        
                        # Receive the JPEG data
                        jpeg_data = camera_socket.recv(jpeg_size, socket.MSG_WAITALL)
                        if not jpeg_data or len(jpeg_data) != jpeg_size:
                            print("Incomplete frame received")
                            break
                        
                        image = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Update the global frame with thread safety
                        with self.frame_lock:
                            self.frame = image
                            
                    except Exception as e:
                        print(f"Camera error: {str(e)}")
                        break
                    
                camera_socket.close()
                
            except Exception as e:
                print(f"Camera connection error: {str(e)}")
                time.sleep(1)
                
                # Reset frame when disconnected
                with self.frame_lock:
                    self.frame = None