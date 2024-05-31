import os
import rasterio
import numpy as np
import scipy.ndimage
from PIL import Image as image
from osgeo import gdal, osr 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

#def main():
tile='14TPP'
dir_name="/gpfs/scratch/khtran/Data/HLS/v1.4/"+tile+'/'

filelist = []
for root,dirs,files in os.walk(dir_name):
	for file in sorted(files):
		filelist.append(os.path.join(root,file))

#concatenate all weekly medians
period = 30 #weekly
medians20 = np.concatenate([mediancomposite(2020,doy_start,period) for doy_start in range (157,367,period)],axis=2) #April 1 doy 92 #Jun 6 - doy 157
medians21 = np.concatenate([mediancomposite(2021,doy_start,period) for doy_start in range (1,121,period)],axis=2) #April 30 doy 120+1

# Preparing median image for the county
# concatenate
# medians_composite = medians21.copy()
medians_composite = np.concatenate((medians20,medians21),axis=2) ##1367, 1876, 1044

# couldy percentage in the entire county
# reshape medians
nbands = 18
nfeatures = medians_composite.shape[2]
imrow = 1367
imcol = 1876
immedian = medians_composite.reshape(imrow*imcol,-1)
imcloud = immedian[(~np.isnan(immedian)).all(axis=1)&(immedian!=-np.inf).all(axis=1)].shape[0]
print('100% clear proportion of median image = '+str(round(imcloud/(imrow*imcol)*100,2)))

# Replace all invalid values by nan value
medians = medians_composite.copy()
medians[medians==-np.inf]=np.nan
medians[medians==np.inf]=np.nan

# fill=32767
# medians[np.isnan(medians)]=fill
# #######_________interpolation_________##########
itp=[]
for x in range(0,nbands): # 18 features
	Bx=np.stack(medians[:,:,i] for i in range(x,nfeatures,nbands))
	Bx_medians=Bx.reshape(int(nfeatures/nbands),-1)
	Bx_df=pd.DataFrame(Bx_medians)
	Bx_medians_interp=Bx_df.interpolate(method='linear', limit_direction='both', axis=0)
	Bx_medians_interp_arr=Bx_medians_interp.to_numpy().reshape(int(nfeatures/nbands),medians.shape[0],-1)#shape 58 x x 
	Bx_interp=np.moveaxis(Bx_medians_interp_arr,0,-1)
	itp.append(Bx_interp)
	print("Band "+ str(x) +"/"+str(nbands)+" done!")
medians_interpolated = np.moveaxis(np.array(itp),0,-1).reshape(imrow,imcol,-1)
medians = medians_interpolated.copy()
# #######________end interpolation________##########

# Write medians to an image BIP file
outdir ='/gpfs/scratch/khtran/2021/Tillage/ML/Training_data/median_im/'
Path(outdir).mkdir(parents=True, exist_ok=True)
medians.reshape(imrow*imcol,-1).tofile(outdir+'monthly_median_im_minnehaha_20_21.bip')

# ######________Read field survey data________########
#top left of Minnehaha county
Yupleft = 4858620
Xupleft = 649860
#Read field survey data

survey_file = '/gpfs/scratch/khtran/2021/Tillage/data/Field_Survey/Tillage_Minnehaha.xls'
fiedsurvey_df = pd.read_excel(survey_file)
tillage = fiedsurvey_df['tillage']
croptypes = fiedsurvey_df['crop']
pX = fiedsurvey_df['X'].values #X coordinate
pY = fiedsurvey_df['Y'].values #Y coordinate

X = [] #Median composition
Y = [] #Tillage
Y1 = [] #Crop types
rows = []
cols = []
for i in range(len(pX)):
	row = round((Yupleft-pY[i])/30) #Y is latitude in arcgis is col
	col = round((pX[i]-Xupleft)/30)
	X.append(medians[row,col])
	Y.append([str(tillage[i])])
	Y1.append([str(croptypes[i])])
	rows.append(row)
	cols.append(col)

X = np.array(X) #Compositions
Y = np.array(Y).flatten() #tillage
Y1 = np.array(Y1).flatten() #crop

rows=np.array(rows).flatten()
cols=np.array(cols).flatten()
###--------Export to Excel file---------###
df_excel = pd.DataFrame()
df_excel['Tillage'] = Y
df_excel['Crop'] = Y1
df_excel['row'] = rows
df_excel['col'] = col
bandnames=['blue','green','red','nir','swir1','swir2','ndvi','evi2','ndti','sti','ndi5','ndi7','crc','sndvi','ndsvi','osavi','vari','gcvi']
for i in range(int(X.shape[1]/len(bandnames))):
	for j in range(len(bandnames)):
		name = bandnames[j]+'_'+str(i)
		df_excel[name]=X[:,i*len(bandnames)+j]
outdir='/gpfs/scratch/khtran/2021/Tillage/ML/Training_data/'
Path(outdir).mkdir(parents=True, exist_ok=True)
df_excel.to_csv(outdir+'Tillage_465_samples_4tills_crop_Minnehaha.csv')



# #####_________Preparing training datasets___________#####
tilltype = ['conventional_till','reduced_till','mulch_till','no_till']
outdir = '/gpfs/scratch/khtran/2021/Tillage/ML/Models/RF/'+str(period)+'/'
Path(outdir).mkdir(parents=True, exist_ok=True)

location = np.stack((rows,cols),axis=1)
location_train=[]
Y_train=[]
location_test=[]
Y_test=[]
split = 0.3 # split 70/30
for i in range(len(tilltype)):
	i+=1 #resever 0 as invalid
	Y_class = [i for x in range(len(Y[Y==tilltype[i-1]]))]
	location_class = location[Y==tilltype[i-1]]
	location_train_class, location_test_class, Y_train_class,Y_test_class = train_test_split(location_class,Y_class,test_size=split,random_state=0)
	
	location_train.append(location_train_class)
	location_test.append(location_test_class)
	Y_train.append(Y_train_class)
	Y_test.append(Y_test_class)

# Merge       
location_train = np.concatenate(np.array(location_train))
Y_train = np.concatenate(np.array(Y_train))
location_test = np.concatenate(np.array(location_test))   
Y_test = np.concatenate(np.array(Y_test))  
# #combine reduced-till and mulch-till to conservational-till
# Y_train[Y_train==3]=2
# Y_test[Y_test==3]=2
# #revise no-till to class 3
# Y_train[Y_train==4]=3
# Y_test[Y_test==4]=3

nfeatures=medians.shape[2]
X_train = []
L_train = []
for i in range(location_train.shape[0]):
	rowc = location_train[i,:][0]
	colc = location_train[i,:][1]
	X_train.append(np.nanmean(medians[rowc-1:rowc+1+1,colc-1:colc+1+1].reshape(-1,nfeatures),axis=0)) #average 3x3 window
	L_train.append(Y_train[i])

X_test = []
L_test = []	
for i in range(location_test.shape[0]):
	rowc = location_test[i,:][0]
	colc = location_test[i,:][1]
	X_test.append(np.nanmean(medians[rowc-1:rowc+1+1,colc-1:colc+1+1].reshape(-1,nfeatures),axis=0)) #average 3x3 window
	L_test.append(Y_test[i])

# Merge       
X_train_all = np.array(X_train)
L_train_all = np.array(L_train)
X_test_all = np.array(X_test)
L_test_all = np.array(L_test)

#remove bad quality samples
X_train = X_train_all[(~np.isnan(X_train_all)).all(axis=1)]
L_train = L_train_all[(~np.isnan(X_train_all)).all(axis=1)]
X_test = X_test_all[(~np.isnan(X_test_all)).all(axis=1)]
L_test = L_test_all[(~np.isnan(X_test_all)).all(axis=1)]

# count samples and plot time series NDTI
bandnames=['Blue','Green','Red','Nir','Swir1','Swir2','NDVI','EVI2','NDTI','STI','NDI5','NDI7','CRC','SNDVI','NDSVI','OSAVI','VARI','GCVI']
nfeatures=medians.shape[2]
# tilltype=['conventional_till','conservational_till','no_till']
for bandid in range(len(bandnames)):
	plt.clf()
	for i in range(len(tilltype)):
		colors = ['r', 'g', 'b', 'y']
		linestyles =  ['-', '--', ':', '-.']
		if bandid<1:
			nsample_train = np.count_nonzero(L_train==i+1)
			nsample_test = np.count_nonzero(L_test==i+1)
			print('nsample '+tilltype[i]+':'+str(nsample_train+nsample_test))
		ax = [4*30+157+k1*period+int(period/2) for k1 in range(0,int(nfeatures/nbands))]
		x_class = np.concatenate((X_train[L_train==i+1],X_test[L_test==i+1]),axis=0)
		aymean =  np.nanmean(x_class,axis=0)
		#bandid = 6 #ndvi
		ayband = [aymean[j] for j in range(bandid,nfeatures,nbands)]
		z=plt.plot(ax,ayband, linestyle=linestyles[i],color=colors[i],label=tilltype[i])
	z=plt.xlabel('DOY')
	if bandid<6:
		z=plt.ylabel('Surface reflectance x 10000')
	else:
		z=plt.ylabel('Indices x 10000')
	z=plt.legend()
	z=plt.title(bandnames[bandid],fontsize = 12,fontweight='bold')
	figdir = '/gpfs/scratch/khtran/2021/Tillage/ML/Training_data/Timeseries/v3/'
	Path(figdir).mkdir(parents=True, exist_ok=True)
	z=plt.savefig(figdir+bandnames[bandid]+'_Timeseries_'+str(period)+'.png')
# #####_________End preparing training datasets___________#####	

############______________Classification_____________################
#RandomForestClassifier
model_RF = RandomForestClassifier()
#model_RF.fit(X_train,L_train)
parameters = {'n_estimators': [100,200,300,400,500],
			'criterion': ['gini','entropy'],
			'max_features': ['auto','sqrt','log2']}
# #Define the grid search object
grid = GridSearchCV(estimator = model_RF,
					param_grid=parameters,
					cv = 5,
					n_jobs=-1) # -1 will ensure that all the cores of the processor is being used in parallel mode
# Fit the grid using train set
grid.fit(X_train, L_train)
print('Best parameters for model:',grid.best_params_)
model_RF = grid.best_estimator_
# model_RF = RandomForestClassifier(criterion= 'gini', max_features= 'auto', n_estimators= 300)
# model_RF.fit(X_train,L_train)
L_pred_train = model_RF.predict(X_train)
print('Model Training accuracy: % .3f' % accuracy_score(L_train, L_pred_train))
print('Model Training kappa: % .3f' % cohen_kappa_score(L_train, L_pred_train))
print('Model Training f-score: % .3f' % f1_score(L_train, L_pred_train, average = 'weighted'))
cm_testing = confusion_matrix(L_train, L_pred_train)
print(cm_testing)
plt.clf()
plt.figure(figsize=(7, 6))
sns.heatmap(cm_testing, cmap = 'jet',annot=True,fmt='d')
tick_marks = np.arange(len(tilltype))+0.5
plt.xticks(tick_marks,tilltype)
plt.yticks(tick_marks,tilltype)
plt.xlabel('Predicted label\nAccuracy={:0.2f}'.format(accuracy_score(L_train, L_pred_train)))
plt.ylabel('True')
#plt.rcParams.update({'font.size': 10})
plt.savefig(figdir+'cm_train_'+str(period)+'bestm.png')

L_pred_test = model_RF.predict(X_test)
print('Model Testing accuracy: % .3f' % accuracy_score(L_test, L_pred_test))
print('Model Testing kappa: % .3f' % cohen_kappa_score(L_test, L_pred_test))
print('Model Testing f-score: % .3f' % f1_score(L_test, L_pred_test, average = 'weighted'))
cm_testing = confusion_matrix(L_test, L_pred_test)
print(cm_testing)
plt.clf()
plt.figure(figsize=(7, 6))
sns.heatmap(cm_testing, cmap = 'jet',annot=True,fmt='d')
tick_marks = np.arange(len(tilltype))+0.5
plt.xticks(tick_marks,tilltype)
plt.yticks(tick_marks,tilltype)
plt.xlabel('Predicted label\nAccuracy={:0.2f}'.format(accuracy_score(L_test, L_pred_test)))
plt.ylabel('True')
#plt.rcParams.update({'font.size': 10})
plt.savefig(figdir+'cm_test_'+str(period)+'bestm.png')

# Whole county classification
immedian = medians_composite.reshape(imrow*imcol,-1)
L_pred_all = best_model_RF.predict(immedian)
im = L_pred_all.reshape(imrow,imcol)

file_CDL= '/gpfs/scratch/khtran/2021/Tillage/data/CDL/CDL_2020_46099.tif'
CDL=rasterio.open(file_CDL).read(1)
im_crs=rasterio.open(file_CDL)
im_clip=im[:CDL.shape[0],:CDL.shape[1]]
masktill = (CDL==1)|(CDL==5)|(CDL==36)|(CDL==28)|(CDL==27)
Till_clip=np.where(masktill==True,im_clip,0)
Till30m = rasterio.open('/gpfs/scratch/khtran/2021/Tillage/Classification/v1/Tillage30m_Minnehaha_2020.tif','w',driver='GTiff',
                         width=im.shape[1], height=im.shape[0],
                         count=1,
                         crs=im_crs.crs,
                         transform=im_crs.transform,
                         dtype='uint8'
                         )
Till30m.write(Till_clip,1)
Till30m.close()

###################Develop different fuctions####################
def yyyydoy(file):
	return(file.split('/')[-1].split('.')[3])

def QAmask(fmask_array,bands_number): #bands_number means the number of bands need to be masked
	bitword_order = (1, 1, 1, 1, 1, 1, 2)  # set the number of bits per bitword
	num_bitwords = len(bitword_order)      # Define the number of bitwords based on your input above
	total_bits = sum(bitword_order)        # Should be 8, 16, or 32 depending on datatype
	qVals = list(np.unique(fmask_array))  # Create a list of unique values that need to be converted to binary and decoded
	all_bits = list()
	goodQuality = []
	for v in qVals:
		all_bits = []
		bits = total_bits
		i = 0
		# Convert to binary based on the values and # of bits defined above:
		bit_val = format(v, 'b').zfill(bits)
		#print('\n' + str(v) + ' = ' + str(bit_val))
		all_bits.append(str(v) + ' = ' + str(bit_val))
		# Go through & split out the values for each bit word based on input above:
		for b in bitword_order:
			prev_bit = bits
			bits = bits - b
			i = i + 1
			if i == 1:
				bitword = bit_val[bits:]
				#print(' Bit Word ' + str(i) + ': ' + str(bitword))
				all_bits.append(' Bit Word ' + str(i) + ': ' + str(bitword)) 
			elif i == num_bitwords:
				bitword = bit_val[:prev_bit]
				#print(' Bit Word ' + str(i) + ': ' + str(bitword))
				all_bits.append(' Bit Word ' + str(i) + ': ' + str(bitword))
			else:
				bitword = bit_val[bits:prev_bit]
				#print(' Bit Word ' + str(i) + ': ' + str(bitword))
				all_bits.append(' Bit Word ' + str(i) + ': ' + str(bitword))
		# 2, 4, 5, 6 are the bits used. All 4 should = 0 if no clouds, cloud shadows were present, and pixel is not snow/ice/water
		if int(all_bits[2].split(': ')[-1]) + int(all_bits[4].split(': ')[-1]) + \
		int(all_bits[5].split(': ')[-1]) + int(all_bits[6].split(': ')[-1]) == 0:
			goodQuality.append(v)
	mask=np.in1d(fmask_array, goodQuality, invert=True).reshape(fmask_array.shape[0],-1)
	mask=np.repeat(mask[None,...],bands_number,axis=0)
	mask=np.moveaxis(mask,0,-1)
	return mask

def prepareHLS(file):
	Minnehaha = [1380,2747,1662,3538]
	hdf_ds = gdal.Open(file, gdal.GA_ReadOnly)
	subdatasets = hdf_ds.GetSubDatasets()
	rawbands=[]
	if (file.split('/')[-1].split('.')[1]=='S30'):
		bands = [1,2,3,8,11,12,13] #Blue Green Red NIR SWIR1 SWIR2
		for i in bands:
			subdataset_name = subdatasets[i][0]
			#print(subdataset_name)
			band_ds = gdal.Open(subdataset_name, gdal.GA_ReadOnly)
			band_array = band_ds.ReadAsArray().astype(np.int16)
			rawbands.append(band_array[Minnehaha[0]:Minnehaha[1],Minnehaha[2]:Minnehaha[3]])
	else: #L30
		bands = [1,2,3,4,5,6,10] #Blue Green Red NIR SWIR1 SWIR2
		for i in bands:
			subdataset_name = subdatasets[i][0]
			#print(subdataset_name)
			band_ds = gdal.Open(subdataset_name, gdal.GA_ReadOnly)
			band_array = band_ds.ReadAsArray().astype(np.int16)
			rawbands.append(band_array[Minnehaha[0]:Minnehaha[1],Minnehaha[2]:Minnehaha[3]])
	bands = np.moveaxis(np.array(rawbands),0,-1) #1367, 1876, 7
	bands = np.where(bands==-1000,np.nan,bands) #mask out all invalida values (without overlap)
	#calculate indices
	scale = 10000
	QA = bands[:,:,6].astype('uint8');
	blue = bands[:,:,0]/scale; green = bands[:,:,1]/scale; red = bands[:,:,2]/scale; nir = bands[:,:,3]/scale;
	swir1 = bands[:,:,4]/scale; swir2 = bands[:,:,5]/scale;
	
	ndvi = (nir-red)/(nir+red)
	evi2 = 2.5*((nir-red)/(nir+2.4*red+1))
	ndti = (swir1-swir2)/(swir1+swir2)
	sti = swir1/swir2
	ndi5 = (nir-swir1)/(nir+swir1)
	ndi7 = (nir-swir2)/(nir+swir2)
	crc = (swir1-green)/(swir1+green)
	sndvi = (nir-red)/(red+nir+0.16)
	ndsvi = (swir1-red)/(swir1 + red)
	osavi = (1 + 0.16)*(nir-red)/(nir+red+0.16)
	vari = (green-red)/(green+red-blue)
	gcvi = (nir/green)-1
	
	#stack all the bands
	bandsstack = np.dstack((blue,green,red,nir,swir1,swir2,ndvi,evi2,ndti,sti,ndi5,ndi7,crc,sndvi,ndsvi,osavi,vari,gcvi))
	
	#define good-quality pixels
	#Call QAmask function
	mask = QAmask(QA,bandsstack.shape[2])
	
	#mask all bands
	bandsstack = np.where(mask==True,np.nan,bandsstack)
	
	return bandsstack*scale #1367, 1876, 18

def mediancomposite(yyyy,doy_start,period):
	weeklystack=[]
	for file in sorted(filelist,key=yyyydoy):
		#extract yyyydoy
		#call yyyydoy function
		doy = yyyydoy(file)[-3:]
		if ((file.endswith('.hdf')) & (str(yyyy) in file) & (int(doy)>=doy_start)& (int(doy)<doy_start+period)):
			print(file)
			#call functions
			bands = prepareHLS(file)
			weeklystack.append(bands)
	#Calculate weekly median
	median = []
	for i in range(bands.shape[2]):
		subband = [weeklystack[x][:,:,i] for x in range (0,len(weeklystack))]
		subband_median = np.nanmedian(subband, axis=0)
		median.append(subband_median)
	#Stack weekly median
	median = np.stack(median,axis=0)
	median = np.moveaxis(median,0,-1)
	print("=======end of this composite from doy "+ str(doy_start) +" to "+str(doy_start+period)+"========")
	return median