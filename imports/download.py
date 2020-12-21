import os

#this script download the AI2020 library from github
def download_AI_library():
	os.system('rm -r AI2020/')
	os.system('git clone https://github.com/UmbertoJr/AI2020.git &> /dev/null')