import train_stacked_ae as t
log = '0Hresultslog.txt'

t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[0],outs=10,resultslog=log)

for i in range(50,500,50):
    t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[0],outs=i,resultslog=log)
#    t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[i],outs=(0.5*i),resultslog=log)
for i in range(500,1000,100):
    t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[0],outs=i,resultslog=log)
    

#t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[100],outs=10,resultslog=log)
'''
t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[300,150],outs=50,resultslog=log)
t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[300,80],outs=50,resultslog=log)
t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[300,200],outs=10,resultslog=log)
t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[300,150],outs=20,resultslog=log)
t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[300,80],outs=30,resultslog=log)
t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[300,50],outs=10,resultslog=log)
t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[600,400],outs=200,resultslog=log)
t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[800,400,100],resultslog=log)
t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[800,500,100],resultslog=log)
'''
reslog = open(log,'a+')
reslog.write('------------------------------------------------------------------------------------------------------')
reslog.close()
