import train_stacked_ae as t
log = '2Hresultslog.txt'

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
reslog = open(log,'a+')
reslog.write('------------------------------------------------------------------------------------------------------')
reslog.close()
