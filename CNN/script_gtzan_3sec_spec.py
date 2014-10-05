import train_CNN as t

logfile = 'CNNresults.txt'
#datasets = t.evaluate_lenet5('../spectrogram/3sec_28x28_gs',input_type='spec',learning_rate=0.1, n_epochs=200,dimx=28,dimy=28,pcancomps=100,nceps=33,nkerns=[20, 50], batch_size=500, results_log=logfile,reinput=None)
datasets = t.evaluate_lenet5('../spectrogram/3sec_28x28_gs',input_type='spec',learning_rate=0.1, n_epochs=100,dimx=28,dimy=28,pcancomps=100,nceps=33,nkerns=[20, 50], batch_size=500, results_log=logfile,reinput=None)
t.evaluate_lenet5('../spectrogram/3sec_28x28_gs',input_type='spec',learning_rate=0.05, n_epochs=200,dimx=28,dimy=28,pcancomps=100,nceps=33,nkerns=[20, 50], batch_size=500, results_log=logfile,reinput=datasets)
t.evaluate_lenet5('../spectrogram/3sec_28x28_gs',input_type='spec',learning_rate=0.005, n_epochs=200,dimx=28,dimy=28,pcancomps=100,nceps=33,nkerns=[20, 50], batch_size=500, results_log=logfile,reinput=datasets)
t.evaluate_lenet5('../spectrogram/3sec_28x28_gs',input_type='spec',learning_rate=0.0005, n_epochs=200,dimx=28,dimy=28,pcancomps=100,nceps=33,nkerns=[20, 50], batch_size=500, results_log=logfile,reinput=datasets)
t.evaluate_lenet5('../spectrogram/3sec_28x28_gs',input_type='spec',learning_rate=0.05, n_epochs=100,dimx=28,dimy=28,pcancomps=100,nceps=33,nkerns=[20, 50], batch_size=500, results_log=logfile,reinput=datasets)
t.evaluate_lenet5('../spectrogram/3sec_28x28_gs',input_type='spec',learning_rate=0.1, n_epochs=200,dimx=28,dimy=28,pcancomps=100,nceps=33,nkerns=[20, 50], batch_size=200, results_log=logfile,reinput=datasets)
t.evaluate_lenet5('../spectrogram/3sec_28x28_gs',input_type='spec',learning_rate=0.1, n_epochs=200,dimx=28,dimy=28,pcancomps=100,nceps=33,nkerns=[20, 50], batch_size=100, results_log=logfile,reinput=datasets)
reslog = open(logfile,'a+')
reslog.write('------------------------------------------------------------------------------------------------------')
reslog.close()

