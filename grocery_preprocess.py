import os
labels = {'cereals':1,'cleaning':2,'beverages':3,'chips+snacks':4,'cannedgoods':5}
dirname = '/data/shared/images/grocerystore/'
files = os.listdir(dirname)
o = open('./grocery_data.labels.txt','w')
for f in files:
	files1 = os.listdir(dirname+f)
	for f1 in files1:
		i = f1.find('.png')
		if i != -1:
			o.write('%s/%s,%d\n'%(f,f1,labels[f]))
o.close()
