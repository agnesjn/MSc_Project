## Training Guideline

"data\_prepare.py" transforms original BiPo and AmBe data into 16bit ndArray and also produces the variance map as the auxiliary data for training. You need to fill in the paths of original data at line 39 and 40 of "data\_prepare.py". The generated data will saved at the place same as this script by default. You can also download the data here and put them directly in the folder of the script.

"model.py" is the file that describes the architecture and forward propagation of the model. As the core code, this file is well-annotated. Most of the innovations of my project are in the function "sal_forward" (line 82 to 134 in "model.py"). Class BiPoCNN is the baseline model for comparing. 

If you have fully cloned the scripts and have set the path of original or pre-processed data, you can start training by running "main.py" (without extra params). The tensorboard log will be saved in the same directory. You can change the max number of epoch at line 41 in file "main.py" 

