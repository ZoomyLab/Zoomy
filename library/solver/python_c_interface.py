import subprocess
import os

def build(architecture = 'cpu'):
   main_dir = os.getenv("SMS")
   path = f"{main_dir}/library/solver"
   # command = ["make", f"arch={architecture}"] 
   command = "make" 
   make_process = subprocess.Popen(command, shell=True, stderr=subprocess.STDOUT, cwd=path)
   if make_process.wait() != 0:
      print(stderr) 
      
def run_driver():
   main_dir = os.getenv("SMS")
   path = f"{main_dir}/bin"
   command = "./volkos"
   make_process = subprocess.Popen(command, shell=True, stderr=subprocess.STDOUT, cwd=path)



if __name__ == "__main__":
   build()
   run_driver()