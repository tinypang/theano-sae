import train_stacked_ae as t
log = 'test_mpc2Hresultslog.txt'

datasets = t.test_SdA(path='./GTZAN_genre',batch_size=20,hidlay=[300,300],outs=50,corruption_levels=[.35, .35],input_type='mpc',nceps=33,resultslog=log,finetune_lr=0.005, pretraining_epochs=100,pretrain_lr=0.0005)
t.test_SdA(path='./GTZAN_genre',batch_size=20,hidlay=[300,300],outs=200,corruption_levels=[.35, .35],input_type='mpc',nceps=33,resultslog=log,finetune_lr=0.005, pretraining_epochs=100,pretrain_lr=0.0005,reinput=datasets)
reslog = open(log,'a+')
reslog.write('------------------------------------------------------------------------------------------------------')
reslog.close()
