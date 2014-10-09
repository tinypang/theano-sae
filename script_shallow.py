from shallow_classify import test_classify as t

#t('./spectrogram/GTZAN_genre/3sec_50x20_gs',10,intype='spec',SVC=True,linSVC=True,RNDforest=True,logR=True,indata=None,inlabels=None)
#t('./spectrogram/GTZAN_genre/full_50x20_gs',10,intype='spec',SVC=True,linSVC=True,RNDforest=True,logR=True,indata=None,inlabels=None)
#t('./spectrogram/GTZAN_genre/3sec_28x28_gs',10,intype='spec',SVC=True,linSVC=True,RNDforest=True,logR=True,indata=None,inlabels=None)
#t('./audio_snippets/GTZAN_3sec',10,intype='mpc',SVC=True,linSVC=True,RNDforest=True,logR=True,indata=None,inlabels=None)
t('./spectrogram/ISMIR_genre/ismirg_3sec_50x20_gs',3,intype='spec',SVC=True,linSVC=True,RNDforest=True,logR=True,indata=None,inlabels=None)
t('./spectrogram/ISMIR_genre/ismirg_3sec_28x28_gs',3,intype='spec',SVC=True,linSVC=True,RNDforest=True,logR=True,indata=None,inlabels=None)
t('./audio_snippets/ismir_genre_training_3sec',3,intype='mpc',SVC=True,linSVC=True,RNDforest=True,logR=True,indata=None,inlabels=None)
