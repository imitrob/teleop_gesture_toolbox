
def time_warp_launch(Xpalm, Y, args=[]):
    ''' Performs time warping to dataset
        - Uses Xpalm trajectory

        After time warp computation:
            - results (id's computed) are compared to Y (id's real)
    Parameters:
        Xpalm (ndarray): Trajectory positions of palm
        Y (1darray): Gesture (id) to record (X_palm)
        args (Str[]): Flags config
            - 'eacheach':
            - 'random':
            - 'promp':
            - 'euclidean':
            - 'crossvalidation':
    '''
    assert len(Xpalm) == len(Y)
    g = list(dict.fromkeys(Y))
    g
    counts = [list(Y).count(g_) for g_ in g]
    counts
    sum(counts)

    paths = Xpalm

    if True:
        if 'eacheach' in args:
            t=time.time()
            # samples x samples
            results = np.zeros([sum(counts),])
            for j in range(0,sum(counts)):
                dist = np.zeros([sum(counts),])
                for i in range(0,sum(counts)):
                    dist[i], _ = fastdtw(paths[j], paths[i])

                mean = np.zeros([len(counts),])
                for i in range(0,len(counts)):
                    mean[i] = np.mean(dist[sum(counts[0:i]):sum(counts[0:i])+counts[i]])
                results[j] = np.argmin(mean)
            print(abs(t-time.time()))
        elif 'random' in args:
            # samples x random
            t = time.time()
            results = np.zeros([sum(counts),])
            for j in range(0,sum(counts)):
                dist = np.zeros([len(counts),])
                for i in range(0,len(counts)):
                    rand_ind = random.randint(0,counts[i]-1)
                    dist[i], _ = fastdtw(paths[j], paths[sum(counts[0:i])], dist=euclidean)

                results[j] = np.argmin(dist)
            print(abs(time.time()-t))

        elif 'promp' in args:
            t = time.time()
            paths = promp_lib2.construct_promp_trajectories(Xpalm, DXpalm, Y)
            paths.shape
            results = np.zeros([sum(counts),])
            for j in range(0,sum(counts)):
                dist = np.zeros([len(counts),])
                for i in range(0,len(counts)):
                    dist[i], _ = fastdtw(Xpalm[j], paths[i], radius=1, dist=euclidean)
                results[j] = np.argmin(dist)
            print(abs(time.time()-t))


        elif 'euclidean' in args:
            paths = promp_lib2.construct_promp_trajectories(Xpalm, DXpalm, Y)
            paths.shape
            results = np.zeros([sum(counts),])
            for j in range(0,sum(counts)):
                dist = np.zeros([len(counts),])
                for i in range(0,len(counts)):
                    dist[i], _ = fastdtw(Xpalm[j], paths[i], dist=euclidean)
                results[j] = np.argmin(dist)

        elif 'crossvalidation' in args:
            # samples x random
            results = np.zeros([sum(counts),])
            for j in range(0,sum(counts)):
                dist = np.zeros([len(counts),])
                for i in range(0,len(counts)):
                    rand_ind = random.randint(0,counts[i]-1)
                    index1 = sum(counts[0:i])
                    index2 = sum(counts[0:i])+rand_ind
                    dist[i], _ = fastdtw(paths[j], paths[index1:index2], dist=euclidean)

                results[j] = np.argmin(dist)

        else: raise Exception("Wrong arg flag")

    name = 'tmp'
    confusion_matrix_pretty_print.plot_confusion_matrix_from_data(Y, results, Gs,
      annot=True, cmap = 'Oranges', fmt='1.8f', fz=10, lw=25, cbar=False, figsize=[6,6], show_null_values=2, pred_val_axis='y', name=name)

    print("Accuracy = {}%".format((Y == results).mean() * 100))
