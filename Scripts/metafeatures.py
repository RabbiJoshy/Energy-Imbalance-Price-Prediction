from Utils.metautils import *
contexts = ['season_cat']
windowsteps = [[96, 24]]#[[96, 24], [96*7, 96]]

argv = [96*7, 96, 2018, 'label', ['mean', 'std', 'kurtosis', 'adf', 'ac1', 'max', 'skew']] #remove argv[2]
fargv = ['ClusteringData/temp.sola.clou.prec.temp.temp.Week/Test', 'ClusteringData/temp.sola.clou.prec.temp.temp.Week/Train']

metadatadf = gettraininmetagdf(fargv[0], argv[0], argv[1], argv[3], argv[4])

for context in contexts:
    print(context)
    for windowstep in windowsteps:
        print(windowstep)
        argv[0] = windowstep[0]
        argv[1] = windowstep[1]
        argv[3] = context
        view = metafeatures(argv,fargv)
        viewo = onehot(argv, fargv)

        sum(view[0] > view[1]) / len(view)



# sum(view[0] > 0.3)/ len(view)
# for column in view.columns:
#     print(column)
# sum(view[0] > view[1])/ len(view)











#
# for i in range(len(W) - window_size + 1):
#     print(W[i: i + window_size]

#do i make consistent wihndows all llikk 1 month or somecs specfic timelength (start with a day)
#so one option is to have a day with a sliding window, another option is to just have each day be a training point


