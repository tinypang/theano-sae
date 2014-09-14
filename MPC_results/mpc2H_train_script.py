import train_stacked_ae as t
log = 'mpc2Hresultslog.txt'

datasets = t.test_SdA(path='./GTZAN_genre',batch_size=20,hidlay=[300,300],outs=50,corruption_levels=[.35, .35],input_type='mpc',nceps=33,resultslog=log,finetune_lr=0.005, pretraining_epochs=100,pretrain_lr=0.0005)
t.test_SdA(path='./GTZAN_genre',batch_size=20,hidlay=[300,300],outs=200,corruption_levels=[.35, .35],input_type='mpc',nceps=33,resultslog=log,finetune_lr=0.005, pretraining_epochs=100,pretrain_lr=0.0005,reinput=datasets)
t.test_SdA(path='./GTZAN_genre',batch_size=20,hidlay=[300,200],outs=100,corruption_levels=[.35, .35],input_type='mpc',nceps=33,resultslog=log,finetune_lr=0.005, pretraining_epochs=100,pretrain_lr=0.0005,reinput=datasets)
t.test_SdA(path='./GTZAN_genre',batch_size=20,hidlay=[400,400],outs=100,corruption_levels=[.35, .35],input_type='mpc',nceps=33,resultslog=log,finetune_lr=0.005, pretraining_epochs=100,pretrain_lr=0.0005,reinput=datasets)
t.test_SdA(path='./GTZAN_genre',batch_size=20,hidlay=[150,150],outs=50,corruption_levels=[.35, .35],input_type='mpc',nceps=33,resultslog=log,finetune_lr=0.005, pretraining_epochs=100,pretrain_lr=0.0005,reinput=datasets)
t.test_SdA(path='./GTZAN_genre',batch_size=20,hidlay=[150,150],outs=50,corruption_levels=[.4, .4],input_type='mpc',nceps=33,resultslog=log,finetune_lr=0.005, pretraining_epochs=100,pretrain_lr=0.0005,reinput=datasets)
t.test_SdA(path='./GTZAN_genre',batch_size=20,hidlay=[150,150],outs=50,corruption_levels=[.2, .2],input_type='mpc',nceps=33,resultslog=log,finetune_lr=0.005, pretraining_epochs=100,pretrain_lr=0.0005,reinput=datasets)
reslog = open(log,'a+')
reslog.write('------------------------------------------------------------------------------------------------------')
reslog.close()
