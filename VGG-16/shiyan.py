
import sys

a = [1, 2.1, 3]
'''
filename = open('F:/PYTHON/dict','w')
filename.write('[')
for k ,v in a.items():
    filename.write(k+':'+str(v)+';')
filename.write(']')
filename.close()
'''
filename = open('F:/PYTHON/VGG-16/train_accuracy_history','w')
for i,j in enumerate(a):
    filename.write(str(i)+' epoch train_acc'+': '+str(j))
    filename.write('\n')
filename.close()
