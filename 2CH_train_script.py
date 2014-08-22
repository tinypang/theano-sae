import train_stacked_ae as t
log = '2CHresultslog.txt'

for i in range(1,6,1):
    t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[300,100],outs=50,corruption_levels=[i/10,i/10,i/10],resultslog=log)
    t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[900,600],outs=300,corruption_levels=[i/10, i/10, i/10],resultslog=log)
    
t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[300,100],outs=50,corruption_levels=[0.5,0.3 ,0.1 ],resultslog=log)
t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[900,600],outs=50,corruption_levels=[0.5, 0.3, 0.1],resultslog=log)

'''
t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[300,200],outs=50,resultslog=log)
t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[300,150],outs=50,resultslog=log)
t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[300,80],outs=50,resultslog=log)
t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[300,200],outs=10,resultslog=log)
t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[300,150],outs=20,resultslog=log)
t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[300,80],outs=30,resultslog=log)
t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[300,50],outs=10,resultslog=log)
t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[600,400],outs=200,resultslog=log)
#t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[800,400,100],resultslog=log)
#t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[800,500,100],resultslog=log)
'''
reslog = open(log,'a+')
reslog.write('------------------------------------------------------------------------------------------------------')
reslog.close()
