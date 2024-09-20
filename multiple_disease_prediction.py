import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu


heart_model=pickle.load(open("/home/bunnys-weapon/ML web apps/savs/heart.sav","rb"))
diabetes_model=pickle.load(open("/home/bunnys-weapon/ML web apps/savs/diabetes_model.sav","rb"))
parkinson_model=pickle.load(open("/home/bunnys-weapon/ML web apps/savs/parkinson_model.sav","rb"))
parkinson_scaler=pickle.load(open("/home/bunnys-weapon/ML web apps/savs/parkinson_scaler.sav","rb"))
diabetes_scaler=pickle.load(open("/home/bunnys-weapon/ML web apps/savs/diabetes_scaler.sav","rb"))


with st.sidebar:
	selected=option_menu("Multiple Disease Prediction",
											["Heart Disease Prediction","Diabetes Prediction","Parkinson Prediction"],
											icons=["heart","activity","person"],
											default_index=0)
											
if selected=="Heart Disease Prediction":
	def pred_heart(inp):
		inp = np.asarray(inp)
		inp = inp.reshape(1, -1)
		ans = heart_model.predict(inp)
		if ans == 0:
		    return "No heart disease"
		else:
		    return "Clear signs of heart disease"
        
        
	st.title("Heart Disease Prediction")
    
    # Input fields
	age = st.text_input("Age")
	sex = st.text_input("Gender")
	cp = st.text_input("Chest Pain")    
	trestbps = st.text_input("Resting Blood Pressure")    
	chol = st.text_input("Serum Cholestrol")    
	fbs = st.text_input("Fasting Blood Sugar")
	restecg = st.text_input("Resting ECG Results")        
	thalach = st.text_input("Maximum Heart Rate Achieved")        
	exang = st.text_input("Exercise-induced Angina")        
	oldpeak = st.text_input("Oldpeak")        
	slope = st.text_input("Slope")
	ca = st.text_input("Number of Major Vessels (0-3) colored by fluoroscopy")
	thal = st.text_input("Thalassemia")
    
    # Prediction and output
	diagnosis = ""
	if st.button("Get Heart Disease Prediction"):
		input_data = [int(age), int(sex), int(cp), int(trestbps), int(chol), int(fbs), int(restecg),
                          int(thalach), int(exang), float(oldpeak), int(slope), int(ca), int(thal)]
		diagnosis = pred_heart(input_data)
    
	st.success(diagnosis)

if selected=="Diabetes Prediction":
	def pred_diabetes(inp):
		inp=np.asarray(inp)
		inp=inp.reshape(1,-1)
		inp=diabetes_scaler.transform(inp)
		pred=diabetes_model.predict(inp)
		if pred[0]==0:
			return "The patient is non diabetic"
		else:
			return "The patient is diabetic"
    		
	st.title("Diabetes Prediction")
	
	Pregnancies=st.text_input("Pregnancies")	
	Glucose=st.text_input("Glucose")	
	BloodPressure=st.text_input("BP")	
	SkinThickness=st.text_input("SkinThickness")	
	Insulin=st.text_input("Insulin")	
	BMI=st.text_input("BMI")	
	DiabetesPedigreeFunction=st.text_input("Diabetes Pedigree Function")	
	Age=st.text_input("Age")
	
	result=""
	if st.button("Diabetes Prediction"):
		inp=[	int(Pregnancies),	int(Glucose),int(	BloodPressure),  int(SkinThickness)	,int(Insulin),	float(BMI)	,float(DiabetesPedigreeFunction)	,int(Age)	]
		result=pred_diabetes(inp)
	st.success(result)
	
	
if selected=="Parkinson Prediction":
	st.title("Parkinsons Disease Prediction")
	
	def pred_parkinson(inp):
		inp=np.asarray(inp)
		inp=inp.reshape(1,-1)
		inp=parkinson_scaler.transform(inp)
		ans=parkinson_model.predict(inp)

		if ans==0:
			return "The person is perfectly healthy"
		elif ans==1:
			return "The person is has Parkinsons Disease"
	
	
	MDVP_Fo_Hz = st.text_input("MDVP:Fo(Hz)")
	MDVP_Fhi_Hz =st.text_input("MDVP:Fhi(Hz)")
	MDVP_Flo_Hz =st.text_input("MDVP:Flo(Hz)")
	MDVP_Jitter_percent = st.text_input("MDVP:Jitter(%)")
	MDVP_Jitter_Abs = st.text_input("MDVP:Jitter(Abs)")
	MDVP_RAP = st.text_input("MDVP:RAP")
	MDVP_PPQ = st.text_input("MDVP:PPQ")
	Jitter_DDP = st.text_input("Jitter:DDP")
	MDVP_Shimmer = st.text_input("MDVP:Shimmer")
	MDVP_Shimmer_dB = st.text_input("MDVP:Shimmer(dB)")
	Shimmer_APQ3 = st.text_input("Shimmer:APQ3")
	Shimmer_APQ5 = st.text_input("Shimmer:APQ5")
	MDVP_APQ = st.text_input("MDVP:APQ")
	Shimmer_DDA = st.text_input("Shimmer:DDA")
	NHR = st.text_input("NHR")
	HNR = st.text_input("HNR")
	RPDE = st.text_input("RPDE")
	DFA = st.text_input("DFA")
	spread1 = st.text_input("spread1")
	spread2 = st.text_input("spread2")
	D2 = st.text_input("D2")
	PPE = st.text_input("PPE")


	answer=""
	if st.button("Parkinsons prediction"):
		inp=[float(MDVP_Fo_Hz),float(MDVP_Fhi_Hz), float(MDVP_Flo_Hz),float(MDVP_Jitter_percent),
		float(MDVP_Jitter_Abs),float(MDVP_RAP),
	  float(MDVP_PPQ),
	  float(Jitter_DDP),float(MDVP_Shimmer),float(MDVP_Shimmer_dB),
	  float(Shimmer_APQ3),float(Shimmer_APQ5),float(MDVP_APQ),
		float(Shimmer_DDA),float(NHR),float(HNR),float(RPDE),float(DFA),float(spread1),float(spread2),float(D2),float(PPE)]

		answer=pred_parkinson(inp)
		
	st.success(answer)
	
	
