import train_stacked_ae as t
log = '2Hresultslog.txt'

datasets = t.test_SdA(path='spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[600,300],outs=100,resultslog=log,pretraining_epochs=100)
t.test_SdA(path='spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[600,300],outs=50,resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=100,input_type='spec',corruption_levels=[.0, .0])
t.test_SdA(path='spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[800,400],outs=100,resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=100,input_type='spec',corruption_levels=[.0, .0])
t.test_SdA(path='spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[800,400],outs=200,resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=100,input_type='spec',corruption_levels=[.0, .0])
t.test_SdA(path='spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[500,100],outs=50,resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=100,input_type='spec',corruption_levels=[.0, .0])
t.test_SdA(path='spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[500,500],outs=100,resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=100,input_type='spec',corruption_levels=[.0, .0])
t.test_SdA(path='spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[200,200],outs=50,resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=100,input_type='spec',corruption_levels=[.0, .0])
t.test_SdA(path='spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[2000,500],outs=200,resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=100,input_type='spec',corruption_levels=[.0, .0])
t.test_SdA(path='spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[2000,500],outs=100,resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=100,input_type='spec',corruption_levels=[.0, .0])
t.test_SdA(path='spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[2000,800],outs=200,resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=100,input_type='spec',corruption_levels=[.0, .0])
reslog = open(log,'a+')
reslog.write('------------------------------------------------------------------------------------------------------')
reslog.close()
