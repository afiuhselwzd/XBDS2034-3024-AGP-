Setup "Age_Gender_Pred_Code.py"
1. Assign correct path to the following files (lines 34-41):
	- faceProto
	- faceModel
	- ageProto
	- ageModel
	- genderProto
	- genderModel
   The pretrained libraries can be found in the "Pretrained Models" folder.
2. Once the age/gender prediction model is run, there will be a video recording that has been created.
3. Place this video recording in the "Vids" folder for "streamlit_integration.py" to use later.


Setup "streamlit_integration.py"
1. In line 22, set the path for the "Vids" folder.
2. Assign correct path to the following files (lines 113-120):
	- faceProto
	- faceModel
	- ageProto
	- ageModel
	- genderProto
	- genderModel
   The pretrained libraries can be found in the "Pretrained Models" folder.
3. In line 141, set path of home image "sample-output.jpg" (optional)
4. Navigate to directory with "streamlit_integration.py" and run the command: 
   streamlit run "streamlit_integration.py"
5. You will be redirected to streamlit website

