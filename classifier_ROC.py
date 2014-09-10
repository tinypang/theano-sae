from import_mpc import test_classify

data, labels = test_classify('./audio_snippets/GTZAN_3sec',2,SVC=False,linSVC=False,RNDforest=True)
test_classify('./audio_snippets/GTZAN_3sec',3,SVC=False,linSVC=False,RNDforest=True,indata=data,inlabels=labels)
test_classify('./audio_snippets/GTZAN_3sec',4,SVC=False,linSVC=False,RNDforest=True,indata=data,inlabels=labels)
test_classify('./audio_snippets/GTZAN_3sec',5,SVC=False,linSVC=False,RNDforest=True,indata=data,inlabels=labels)
test_classify('./audio_snippets/GTZAN_3sec',2,SVC=True,linSVC=False,RNDforest=False,indata=data,inlabels=labels)
test_classify('./audio_snippets/GTZAN_3sec',2,SVC=False,linSVC=True,RNDforest=False,indata=data,inlabels=labels)
