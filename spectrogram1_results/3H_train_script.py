import train_stacked_ae as t
log = '3Hresultslog.txt'

t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[600,300,100],outs=20,resultslog=log)
t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[600,300,100],outs=40,resultslog=log)
t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[600,300,100],outs=60,resultslog=log)
t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[600,300,100],outs=80,resultslog=log)
t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[600,300,200],outs=100,resultslog=log)
#t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[800,150,100],resultslog=log)
#t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[800,200,100],resultslog=log)
#t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[800,300,100],resultslog=log)
#t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[800,400,100],resultslog=log)
#t.test_SdA(path='spectrogram/gs_full_spectrograms_50th',batch_size=20,hidlay=[800,500,100],resultslog=log)

