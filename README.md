## Training Guideline

*data\_prepare.py* transforms original BiPo and AmBe data into 16bit ndArray and also produces the variance map as the auxiliary data for training. You need to **fill in the paths** of original data at line 39 and 40 of *data\_prepare.py*. The generated data will saved at the place same as this script by default. You can also **download the data** [here](https://drive.google.com/file/d/1IyxCEBWta744_XV84O4yfW-ESO1oDZXT/view?usp=sharing) and put them directly in the folder of the script. Same with the original code, here I use 2/3 data for training and 1/3 for testing. 

*model.py* is the file that describes the architecture and forward propagation of the model. As the core code, this file is well-annotated. Most of the innovations of my project are in the function *sal_forward* (line 82 to 134 in *model.py*). Class BiPoCNN is the baseline model for comparing. *saliency_map.py* is just to clarify how the saliency map is generated, not used as a part of module in the implementation of the method. 

If you have fully cloned the scripts and have set the path of original or pre-processed data, you can start training by running "main.py" (without extra params). The tensorboard log will make a direction ./run in the same directory. You can change the max number of epoch at line 41 in *main.py*.

Below is the result for one test using my full method. The validation accuracy at 24 epoch reaches 0.887. The outcomes of different runs do not vary much.
![avatar](https://github.com/agnesjn/MSc_Project/blob/master/result.png)

I am sorry for not having enough time to put all the hyper-parameters in a configuration file. Most model-related params are in *model.py* and others regarding training setup are in *train.py*. I am happy to help if you want to have a clearer understanding of the parameter settings or try other combinations. Also please let me know if you have any questions about my work.