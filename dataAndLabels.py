import readDataFiles
import glob
import numpy as np


def dataAndLabels():
	dataPath = glob.glob('data/Balgrist_20170508/first/*.csv')
	files2Read = list(set([dataPath[f][29:len(dataPath[f])-6] for f in range(0,len(dataPath))]))
	data={}
	data['Ascend']={}
	data['Descend']={}
	data['Level']={}
	all_data=[]
	labels=[]
	for r in range(0,len(files2Read)):
	    if "Ascend" in files2Read[r]:
	        data['Ascend'][files2Read[r]]={}    
	        data['Ascend'][files2Read[r]]['dataDict']=readDataFiles.actualReading(files2Read[r])
	        if len(data['Ascend'][files2Read[r]]['dataDict'])>2:
	            all_data.append(data['Ascend'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp1"])
	            labels.append(1)
	            all_data.append(data['Ascend'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp2"])
	            labels.append(1)
	        else:
	            all_data.append(data['Ascend'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp1"])
	            labels.append(1)
	            
	    elif "Descend" in files2Read[r]:
	        data['Descend'][files2Read[r]]={}    
	        data['Descend'][files2Read[r]]['dataDict']=readDataFiles.actualReading(files2Read[r])
	        if len(data['Descend'][files2Read[r]]['dataDict'])>2:
	            all_data.append(data['Descend'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp1"])
	            labels.append(2)
	            all_data.append(data['Descend'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp2"])
	            labels.append(2)
	        else:
	            all_data.append(data['Descend'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp1"])
	            labels.append(2)
	    elif "Level" in files2Read[r]:
	        data['Level'][files2Read[r]]={}    
	        data['Level'][files2Read[r]]['dataDict']=readDataFiles.actualReading(files2Read[r])
	        if len(data['Level'][files2Read[r]]['dataDict'])>2:
	            all_data.append(data['Level'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp1"])
	            labels.append(3)
	            all_data.append(data['Level'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp2"])
	            labels.append(3)
	        else:
	            all_data.append(data['Level'][files2Read[r]]['dataDict'][files2Read[r]+"_Exp1"])
	            labels.append(3)
	            
	dataset = all_data[0]
	labels_dataset = np.ones((dataset.shape[0], 1))*labels[0]
	for i in range(1,len(all_data)):
	    dataset = np.append(dataset,all_data[i],axis=0)
	    labels_dataset = np.append(labels_dataset,np.ones((all_data[i].shape[0], 1))*labels[i],axis=0)

	return all_data, labels, data, dataset, labels_dataset


