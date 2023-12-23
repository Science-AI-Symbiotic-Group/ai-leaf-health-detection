import os
import argparse
import subprocess
import sys
import shutil

def install_Libraries():
    print("INFO: INSTALLING LIBRARIES")
    subprocess.call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

def add_scripts_to_path(webcam_file_name,image_file_name):
    platform = sys.platform
    
    if platform == "linux" or platform == "linux2":
        # For linux
        print(f"INFO: You are on platform - LINUX")
        shutil.move("/cv/webcam_keras.py",f"~/usr/local/bin/{webcam_file_name}")
        shutil.move("/cv/image_keras.py",f"../../usr/local/bin/{image_file_name}")
        print('INFO: MOVED ALL FILES TO "usr/local/bin"')
        pass
    elif platform == "darwin":
        # For Mac, NEED TO BE WORKED ON
        print(f"INFO: You are on platform - MACOS")
        pass
    elif platform == "win32":
        # For Windows, NEED TO BE WORKED ON
        print(f"INFO: You are on platform - WINDOWS")
        pass
        

argument_parser = argparse.ArgumentParser()

argument_parser.add_argument("--add_to_path",help="Add this flag to add the script to path",action="store_true")
argument_parser.add_argument("--webcam_file_name",help="Replace the File name for the webcam_keras.py file.")
argument_parser.add_argument("--image_file_name",help="Replace the File name for the image_keras.py file.")




arguments = argument_parser.parse_args()


add_to_path = arguments.add_to_path
webcam_file_name = str(arguments.webcam_file_name)
image_file_name = str(arguments.image_file_name)

if webcam_file_name.endswith(".py"):
    pass
elif webcam_file_name == None or webcam_file_name == "None":
    print("ERROR: Please Give a Name for the webcam_file_name argument")
    exit()
else:
    webcam_file_name = f"{webcam_file_name}.py"

if image_file_name.endswith(".py"):
    pass
elif image_file_name == None or image_file_name == "None":
    print("ERROR: Please Give a Name for the image_file_name argument")
    exit()
else:
    image_file_name = f"{image_file_name}.py"

print(f"INFO: Live Webcam file beings saved as {webcam_file_name}")
print(f"INFO: Image file beings saved as {image_file_name}")



if add_to_path == False:
    print("INFO: NOT ADDING SCRIPTS TO SYSTEM PATH.")
    yes_or_no = input("Would you like to continue? [Y/n]: ")

    yes_or_no = yes_or_no.lower()

    if yes_or_no == "" or yes_or_no == "y":
        install_Libraries()
        print("INFO: INSTALLED ALL DEPENDENCIES SUCCESFULLY")
    else:
        exit()
else:
    print("INFO: ADDING SCRIPTS TO SYSTEM PATH.")
    yes_or_no = input("Would you like to continue? [Y/n]: ")

    yes_or_no = yes_or_no.lower()

    if yes_or_no == "" or yes_or_no == "y":
        #install_Libraries()
        add_scripts_to_path(webcam_file_name=webcam_file_name,image_file_name=image_file_name)
    else:
        exit()




