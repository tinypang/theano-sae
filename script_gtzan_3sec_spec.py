import train_stacked_ae as t
log = 'results_all.txt'

#datasets = t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[2000,500,100],outs=50,corruption_levels=[.1, .2,.3],resultslog=log,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
datasets = t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[2000,1000,500],outs=100,corruption_levels=[.1, .2, .3],resultslog=log,reinput=None,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[2000,1000,500],outs=100,corruption_levels=[.1, .1, .1],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[2000,1000,500],outs=100,corruption_levels=[.2, .2, .2],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[2000,1000,500],outs=100,corruption_levels=[.3, .3, .3],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[2000,1000,500],outs=200,corruption_levels=[.1, .2, .3],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[1000,500,200],outs=50,corruption_levels=[.1, .2, .3],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[800,600,200],outs=50,corruption_levels=[.1, .2, .3],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[800,600,200],outs=100,corruption_levels=[.1, .2, .3],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[600,300,150],outs=100,corruption_levels=[.1, .2, .3],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[600,300,100],outs=50,corruption_levels=[.1, .2, .3],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[500,250,125],outs=60,corruption_levels=[.1, .2, .3],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[600,300,100],outs=50,corruption_levels=[.3, .2, .1],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[600,300,100],outs=50,corruption_levels=[.0, .0, .0],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[600,300,100],outs=50,corruption_levels=[.1, .1, .1],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[600,300,100],outs=50,corruption_levels=[.2, .2, .2],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[600,300,100],outs=50,corruption_levels=[.3, .3, .3],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[600,300,100],outs=50,corruption_levels=[.35, .35, .35],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[600,300,100],outs=50,corruption_levels=[.1, .1, .35],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[600,300],outs=100,corruption_levels=[.1, .1,],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[600,300],outs=50,corruption_levels=[.1, .2],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[500,100],outs=50,corruption_levels=[.2, .3],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[500,100],outs=50,corruption_levels=[.35, .35],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[800,400],outs=200,corruption_levels=[.1, .2],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[800,400],outs=100,corruption_levels=[.1, .2],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[300,100],outs=50,corruption_levels=[.1, .2],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[300],outs=50,corruption_levels=[.1],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[600],outs=300,corruption_levels=[.1],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[600],outs=100,corruption_levels=[.1],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[500],outs=100,corruption_levels=[.1],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[500],outs=100,corruption_levels=[.2],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[500],outs=100,corruption_levels=[.3],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[500],outs=100,corruption_levels=[.35],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
t.test_SdA(path='./spectrogram/3sec_50x20_gs',batch_size=20,hidlay=[100],outs=50,corruption_levels=[.2],resultslog=log,reinput=datasets,dimx=50,dimy=20,pretraining_epochs=25,input_type='spec',finetune_lr=0.1,pretrain_lr=0.001)
reslog = open(log,'a+')
reslog.write('------------------------------------------------------------------------------------------------------')
reslog.close()