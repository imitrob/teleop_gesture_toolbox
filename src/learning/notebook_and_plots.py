import matplotlib.pyplot as plt
import numpy as np
"""
PLOT FUNCTION
"""
def plot(data, title="", xlabel="", ylabel="", legend=[""], subtitle="", rewrite_lower_boundaries=None, minmaxboundaries4plot=False):
    ''' Plots the data, where data[0] is time series and data[1:] are value series
    Parameters:
        data (2darray)
        title (Str): Suptitle
        xlabel, ylabel (Str): Axes titles
        legend (Str[]): Legend titles
        subtitle (Str): Further plot description & notes
        rewrite_lower_boundaries (Float): Sets y axis minimum number in plot
        minmaxboundaries4plot (Bool): If True, 3 rows in data for each series defining (avg, min, max)
            - e.g. [time, series1_average, series1_min, series1_max, series2_average, series2_min, series2_max, ...]
    '''
    plt.rcParams.update({'font.size': 15})
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.set_size_inches(18.5, 7.5)
    colors = ['blue', 'red', 'yellow', 'green', 'grey', 'cyan', 'black']
    if minmaxboundaries4plot:
        for n,i in enumerate(range(1, len(data), 3)):
            ax1.plot(data[0], data[i], lw=2)
            j = data[i-1]
            if i == 1: j = min(data[1])
            if rewrite_lower_boundaries: j=rewrite_lower_boundaries
            ax2.fill_between(data[0], data[i+1], data[i+2], facecolor=colors[n], alpha=0.5)
    else:
        for n,i in enumerate(range(1, len(data))):
            ax1.plot(data[0], data[i], lw=2)
            j = data[i-1]
            if i == 1: j = min(data[1])
            if rewrite_lower_boundaries: j=rewrite_lower_boundaries
            ax2.fill_between(data[0], j, data[i], facecolor=colors[n], alpha=0.5)

    for ax in ax1, ax2:
        ax.grid(True)
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel(xlabel)
    ax1.legend(legend)
    for label in ax2.get_yticklabels():
        label.set_visible(False)

    fig.suptitle(title)
    ax2.set_xlabel(subtitle)


def make_data_boundaries(data_test):
    ''' It takes data_test and computes, mean_values and low/high boundaries, which are given by +-1 sigma (standard deviation)
    Parameters:
        data_test (2darray): [(time + n*m)]
    Returns:
        data_test (2darray): [(time, mean_values1, low_boundaries1, high_boundaries1, mean_values2, low_boundaries2, ...)]
    '''
    low_boundaries, high_boundaries, mean_values = [], [], []
    for values in data_test[1:].T:
        low_boundaries.append(np.mean(values) - np.std(values))
        high_boundaries.append(np.mean(values) + np.std(values))
        mean_values.append(np.mean(values))
    return np.vstack([data_test[0], mean_values, low_boundaries, high_boundaries])

"""
Ability to reproduce in following steps:
    1. Install mirracle_gestures package
    2. Download dataset from gdrive TODO:
    3. Run script from src/learning: `python3.7 train.py`
        - Specify given experiment from class 'Experiments': 'update', 'all',
        - For seed independent measurement, use seed_wrapper function
        - Results can be plotted here

The Balanced accuracy is used: BA (Balanced accuracy) = (TPR (True Positive Rate) + TNR (True Negative Rate)) / 2
Static gestures used: train.Gs = ['grab', 'pinch', 'point', 'respectful', 'spock', 'rock', 'victory']
Observed variables chosen set: all_defined - 87 hand variables
Use more samples from one recording: take sample every n time-sample (take_every_n)
Learn used iterations (iter), Update learning iterations ()
"""

N = 1
TITLE = 'Experiment '+str(N)+', Grid search (Iterations) (Seed independent)'
ARGS = '7 static gestures, all_defined, take_every=10, 1s, interpolate, iter=<1000-4000>, split=0.25 (tested on 75%), n_hidden=50'

data_train = np.array([[1000.        , 1500.        , 1750.        , 2000.        ,        2250.        , 2500.        , 3000.        , 3500.        ,        4000.        ],       [  92.03406814,   94.63927856,   95.24048096,   95.79158317,          95.94188377,   96.29258517,   96.79358717,   96.54308617,          95.39078156],       [  81.36272545,   89.77955912,   90.88176353,   91.48296593,          91.48296593,   91.53306613,   90.13026052,   89.52905812,          88.22645291],       [  91.38276553,   94.98997996,   95.44088176,   95.89178357,          95.44088176,   95.79158317,   94.93987976,   94.68937876,          94.23847695],       [  88.17635271,   90.93186373,   91.53306613,   91.83366733,          92.33466934,   92.48496994,   89.17835671,   88.77755511,          91.18236473],       [  81.41282565,   86.57314629,   87.9759519 ,   88.47695391,          88.0761523 ,   87.7254509 ,   88.0260521 ,   87.2745491 ,          87.2745491 ]])
data_test = np.array([[1000.        , 1500.        , 1750.        , 2000.        ,        2250.        , 2500.        , 3000.        , 3500.        ,        4000.        ],       [  91.59159159,   94.89489489,   95.1951952 ,   96.3963964 ,          96.3963964 ,   97.14714715,   97.2972973 ,   97.14714715,          96.3963964 ],       [  79.12912913,   90.24024024,   90.99099099,   91.89189189,          91.74174174,   91.89189189,   91.44144144,   90.69069069,          89.03903904],       [  88.88888889,   92.49249249,   93.09309309,   94.59459459,          94.59459459,   94.59459459,   94.59459459,   94.29429429,          94.29429429],       [  88.28828829,   90.24024024,   91.44144144,   90.99099099,          90.69069069,   90.69069069,   88.88888889,   87.98798799,          89.33933934],       [  80.78078078,   85.88588589,   87.08708709,   87.68768769,          86.63663664,   86.93693694,   86.93693694,   87.08708709,          85.73573574]])
data_test = make_data_boundaries(data_test)
data_train = make_data_boundaries(data_train)
plot(np.r_[ data_test, data_train[1:]], TITLE+'\nArgs: '+ARGS, 'Iterations [-]', 'Balanced Accuracy [%]', ['test data', 'train data'], minmaxboundaries4plot=True)


N+=1
TITLE = 'Experiment '+str(N)+', Grid search (Iterations) (Seed independent)'
ARGS = '7 static gestures, all_defined, take_every_10, 1s, 0.25split, 10000samples, tested on remaining data, iter=<1000,6000>'

accuracies = [[    1000,     1500,    1750,    2000,    2250,    2500,    3000,    4000,    6000]]
accuracies_train = [[    1000,    1500,    1750,    2000,    2250,    2500,    3000,    4000,    6000]]
accuracies.append([78.79464285714286, 86.60714285714286, 90.625, 86.83035714285714, 91.51785714285714, 94.86607142857143, 87.27678571428571, 96.42857142857143, 94.41964285714286])
accuracies_train.append([78.15063385533185, 89.56002982848621, 91.34973900074571, 88.21774794929156, 92.46830723340791, 94.7054436987323, 91.20059656972408, 95.3765846383296, 95.07829977628636])
accuracies.append([83.25892857142857, 89.50892857142857, 89.0625, 88.61607142857143, 89.73214285714286, 85.71428571428571, 89.95535714285714, 90.84821428571429, 93.75])
accuracies_train.append([83.07233407904549, 91.64802386278896, 91.34973900074571, 91.2751677852349, 92.39373601789708, 85.83146905294556, 87.54660700969426, 94.92915734526473, 94.2580164056674])
accuracies.append([79.01785714285714, 91.51785714285714, 87.27678571428571, 94.86607142857143, 91.51785714285714, 86.38392857142857, 88.61607142857143, 85.71428571428571, 85.26785714285714])
accuracies_train.append([79.19463087248322, 91.42431021625652, 87.69574944071589, 93.21401938851604, 93.51230425055928, 87.69574944071589, 92.54287844891871, 91.8717375093214, 85.68232662192393])
accuracies.append([84.15178571428571, 88.83928571428571, 92.85714285714286, 90.84821428571429, 89.0625, 89.95535714285714, 94.86607142857143, 92.41071428571429, 89.0625])
accuracies_train.append([85.30947054436987, 87.32289336316182, 91.72259507829978, 91.64802386278896, 89.26174496644296, 90.1565995525727, 94.1834451901566, 92.46830723340791, 91.20059656972408])
accuracies.append([84.375, 86.16071428571429, 91.51785714285714, 93.08035714285714, 89.95535714285714, 89.95535714285714, 91.29464285714286, 95.53571428571429, 95.75892857142857])
accuracies_train.append([83.51976137211037, 89.70917225950782, 89.41088739746458, 92.39373601789708, 90.23117076808353, 94.03430275913497, 91.20059656972408, 96.04772557792693, 95.3765846383296])

data_test = np.array(accuracies)
data_train = np.array(accuracies_train)

array_iter10000 = [10000, 89.28571428571429, 92.63392857142857, 91.96428571428571, 87.05357142857143, 87.94642857142857]
array_train_iter10000 = [10000, 93.36316181953765, 91.94630872483222, 90.1565995525727, 86.95003728560775, 89.63460104399702]
array_iter20000 =  [20000, 84.82142857142857, 83.03571428571429, 85.49107142857143, 84.82142857142857, 76.33928571428571]
array_train_iter20000 = [20000, 88.59060402684564, 85.53318419090232, 85.4586129753915, 83.22147651006712, 76.43549589858316]
array_iter30000 = [30000, 81.91964285714286, 81.91964285714286, 89.0625, 85.9375, 72.99107142857143]
array_train_iter30000 = [30000, 85.30947054436987, 83.22147651006712, 87.1737509321402, 82.32662192393735, 72.78150633855331]
array_iter40000 = [40000, 81.47321428571429, 80.13392857142857, 87.27678571428571, 81.47321428571429, 71.42857142857143]
array_train_iter40000 = [40000, 84.19090231170769, 82.32662192393735, 85.30947054436987, 82.5503355704698, 70.54436987322893]


data_test = np.c_[data_test, array_iter10000, array_iter20000, array_iter30000, array_iter40000 ]
data_train = np.c_[data_train, array_train_iter10000, array_train_iter20000, array_train_iter30000, array_train_iter40000 ]


data_test = make_data_boundaries(data_test)
data_train = make_data_boundaries(data_train)
test_previous = data_test[:,0:9]
train_previous = data_train[:,0:9]

plot(np.r_[ data_test, data_train[1:]], TITLE+'\nArgs: '+ARGS, 'Iterations <1000,6000> [-]', 'Balanced Accuracy [%]', ['test data', 'train data'], minmaxboundaries4plot=True)


N+=1
TITLE = 'Experiment '+str(N)+', Grid search (Iterations) with interpolate (Seed independent)'
ARGS = '7 static gestures, all_defined, take_every_10, interpolate, 1s, 0.25split (tested on 25% data), 10000samples, iter=<1000,6000>'

accuracies = [[    1000,     1500,    1750,    2000,    2250,    2500,    3000,    4000,    6000]]
accuracies_train = [[    1000,    1500,    1750,    2000,    2250,    2500,    3000,    4000,    6000]]
accuracies.append([91.5915915915916, 89.93993993993993, 83.63363363363364, 90.84084084084084, 89.93993993993993, 93.84384384384384, 92.94294294294293, 94.14414414414415, 84.23423423423422])
accuracies_train.append([91.53306613226452, 90.7314629258517, 83.36673346693387, 91.93386773547094, 90.98196392785572, 92.68537074148297, 94.58917835671342, 95.19038076152304, 83.41683366733467])
accuracies.append([78.97897897897897, 86.63663663663664, 90.990990990991, 90.990990990991, 91.44144144144144, 91.8918918918919, 94.5945945945946, 90.84084084084084, 92.94294294294293])
accuracies_train.append([81.11222444889779, 89.62925851703407, 92.03406813627254, 90.93186372745491, 91.13226452905812, 93.08617234468937, 93.3867735470942, 93.78757515030061, 89.82965931863728])
accuracies.append([88.88888888888889, 88.73873873873875, 88.43843843843844, 84.53453453453453, 91.29129129129129, 92.34234234234235, 91.74174174174175, 89.4894894894895, 90.39039039039038])
accuracies_train.append([91.03206412825652, 90.83166332665331, 88.72745490981964, 86.17234468937876, 92.43486973947896, 92.43486973947896, 92.58517034068137, 89.17835671342685, 88.77755511022045])
accuracies.append( [88.28828828828829, 88.88888888888889, 90.69069069069069, 91.74174174174175, 93.24324324324324, 96.54654654654654, 93.09309309309309, 91.8918918918919, 90.990990990991] )
accuracies_train.append( [88.72745490981964, 89.32865731462925, 93.78757515030061, 92.73547094188376, 93.687374749499, 95.54108216432866, 91.73346693386773, 93.9879759519038, 88.97795591182364] )
accuracies.append( [80.78078078078079, 84.98498498498499, 91.8918918918919, 93.993993993994, 94.29429429429429, 94.8948948948949, 90.39039039039038, 93.3933933933934, 90.54054054054053] )
accuracies_train.append( [81.312625250501, 84.3687374749499, 91.23246492985973, 93.48697394789579, 94.68937875751503, 94.93987975951904, 90.5811623246493, 93.08617234468937, 90.98196392785572] )

data_test = np.array(accuracies)
data_train = np.array(accuracies_train)
data_test = make_data_boundaries(data_test)
data_train = make_data_boundaries(data_train)

plot(np.r_[ data_test, data_train[1:] ], TITLE+'\nArgs: '+ARGS, 'Iterations <1000,6000> [-]', 'Balanced Accuracy [%]', ['test data', 'train data', '/wo interp test', '/wo interp train'], minmaxboundaries4plot=True)

plot(np.r_[ data_test, test_previous[1:] ], TITLE+'\nArgs: '+ARGS, 'Iterations <1000,6000> [-]', 'Balanced Accuracy [%]', ['test data with interpolation', 'test data', '/wo interp test', '/wo interp train'], minmaxboundaries4plot=True)


N+=1
TITLE = 'Experiment '+str(N)+', (Right Example of weight initialization) 90% trained, updated with 10% data, only 5 update iterations (Seed independent)'
ARGS = 'all_defined, take_every_10, 1s, 0.25split, 2500iter, tested on 25% data, 2500iupdate'
accuracies = [[2500, 2505 ]]
accuracies_train = [[ 2500, 2505 ]]
accuracies.extend( [[91.80633147113593, 91.80633147113593], [90.87523277467412, 91.06145251396647], [90.31657355679702, 90.68901303538175], [94.97206703910615, 95.15828677839852], [93.1098696461825, 93.1098696461825]] )
accuracies_train.extend( [[93.07282415630551, 93.07282415630551], [92.71758436944938, 92.53996447602132], [91.82948490230906, 92.00710479573712], [95.11545293072824, 94.84902309058614], [92.27353463587922, 92.62877442273535]] )

data_test = np.array(accuracies)
data_train = np.array(accuracies_train)
data_test = make_data_boundaries(data_test)
data_train = make_data_boundaries(data_train)

plot(np.r_[ data_test, data_train[1:] ], TITLE+'\nArgs: '+ARGS, 'Iterations (Update) [-]', 'Balanced Accuracy [%]', ['test data', 'train data'], minmaxboundaries4plot=True)

N+=1
TITLE = 'Experiment '+str(N)+', (Right) 90% trained, updated with 10% data, 2500 update iterations (Reason decrease is overtraining with new data) (Seed independent)'
ARGS = 'all_defined, take_every=10, 1s, split=0.25 (75% train, 25% test data), iter=2500, iupdate=2500'

accuracies = [[90, 100 ]]
accuracies_train = [[ 90, 100 ]]
accuracies.extend( [[91.80633147113593, 90.5027932960894], [90.87523277467412, 91.43389199255121], [90.68901303538175, 87.52327746741155], [94.78584729981378, 85.6610800744879], [92.92364990689012, 75.97765363128491]] )
accuracies_train.extend( [[93.07282415630551, 92.36234458259325], [92.71758436944938, 92.71758436944938], [92.00710479573712, 88.27708703374778], [95.0266429840142, 84.28063943161635], [91.82948490230906, 74.1563055062167]] )

data_test = np.array(accuracies)
data_train = np.array(accuracies_train)
data_test = make_data_boundaries(data_test)
data_train = make_data_boundaries(data_train)

plot(np.r_[ data_test, data_train[1:]], TITLE+'\nArgs: '+ARGS, 'proportion updated train data [%]', 'Balanced Accuracy [%]', ['test data', 'train data'], minmaxboundaries4plot=True)




N+=1
TITLE = 'Experiment '+str(N)+', (Right) Sample records for every gesture k=[1,2,5,10,20] (Seed independent)'
ARGS = '7 static gestures, all_defined, take_every_10, interpolate, 1s, 0.3split, 3000iter, tested on 30% data, 2000iupdate'

accuracies = [[1, 2, 5, 10, 20, 50 ]]
accuracies_train = [[ 1, 2, 5, 10, 20, 50 ]]
accuracies.extend( [[33.166458072590736, 63.07884856070088, 77.97246558197747, 85.73216520650814, 88.6107634543179, 91.36420525657071], [37.42177722152691, 56.69586983729662, 82.47809762202753, 83.97997496871089, 90.61326658322903, 92.86608260325406], [55.819774718397994, 72.5907384230288, 79.47434292866082, 89.98748435544431, 91.73967459324155, 89.48685857321652], [27.909887359198997, 55.694618272841055, 81.72715894868585, 89.23654568210263, 83.60450563204004, 91.73967459324155], [44.30538172715895, 64.70588235294117, 67.08385481852316, 84.35544430538174, 87.73466833541927, 88.36045056320401]] )
accuracies_train.extend( [[85.71428571428571, 100.0, 85.71428571428571, 100.0, 100.0, 100.0], [85.71428571428571, 100.0, 100.0, 85.71428571428571, 85.71428571428571, 85.71428571428571], [71.42857142857143, 100.0, 100.0, 100.0, 100.0, 85.71428571428571], [71.42857142857143, 71.42857142857143, 100.0, 100.0, 85.71428571428571, 85.71428571428571], [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]] )

data_test = np.array(accuracies)
data_train = np.array(accuracies_train)
data_test = make_data_boundaries(data_test)
data_train = make_data_boundaries(data_train)

plot(np.r_[ data_test, data_train[1:]], TITLE+'\nArgs: '+ARGS, 'Number samples of each gesture, train data [-]', 'Balanced Accuracy [%]', ['test data', 'train data'], minmaxboundaries4plot=True)

N+=1
TITLE = 'Experiment '+str(N)+', Sample records for every gesture k=[1,2,5,10,20] (Seed independent)'
ARGS = '7 static gestures, all_defined, take_every_10, interpolate, 1s, 0.3split, 3000iter, tested on 30% data, 3000iupdate'

accuracies = [[1, 2, 5, 10, 20, 50 ]]
accuracies_train = [[ 1, 2, 5, 10, 20, 50 ]]
accuracies.extend( [[34.79349186483104, 66.83354192740926, 80.22528160200251, 87.60951188986232, 90.4881101376721, 92.11514392991239], [38.67334167709637, 63.20400500625782, 82.47809762202753, 80.85106382978722, 88.73591989987484, 91.36420525657071], [55.819774718397994, 73.71714643304131, 85.48185231539425, 90.61326658322903, 92.74092615769712, 90.11264080100125], [27.909887359198997, 61.57697121401752, 84.98122653316645, 89.61201501877348, 87.85982478097623, 91.8648310387985], [45.056320400500624, 53.44180225281602, 78.09762202753441, 86.4831038798498, 88.73591989987484, 88.48560700876095]] )
accuracies_train.extend( [[85.71428571428571, 100.0, 100.0, 100.0, 100.0, 100.0], [85.71428571428571, 85.71428571428571, 100.0, 85.71428571428571, 100.0, 100.0], [71.42857142857143, 100.0, 100.0, 100.0, 100.0, 85.71428571428571], [71.42857142857143, 71.42857142857143, 100.0, 100.0, 85.71428571428571, 100.0], [100.0, 85.71428571428571, 100.0, 100.0, 100.0, 100.0]] )


data_test = np.array(accuracies)
data_train = np.array(accuracies_train)
data_test = make_data_boundaries(data_test)
data_train = make_data_boundaries(data_train)

plot(np.r_[ data_test, data_train[1:]], TITLE+'\nArgs: '+ARGS, ' updated train data [-]', 'Balanced Accuracy [%]', ['test data', 'train data'], minmaxboundaries4plot=True)

N+=1
TITLE = 'Experiment '+str(N)+', (Right) 17% initial data 17% update (Seed independent)'
ARGS = '7 static gestures, all_defined, take_every_10, 1s, 0.3split (tested on 30% data), 3000iter, 3000iupdate'

accuracies = [[ 17.5, 35, 52.5, 70 ]]
accuracies_train = [[ 17.5, 35, 52.5, 70 ]]
accuracies.extend( [[93.61702127659575, 94.49311639549437, 93.4918648310388, 89.86232790988737], [88.48560700876095, 93.86733416770964, 90.36295369211514, 90.11264080100125], [91.73967459324155, 91.73967459324155, 90.23779724655819, 87.85982478097623], [92.74092615769712, 88.36045056320401, 87.85982478097623, 88.73591989987484], [89.73717146433042, 90.11264080100125, 88.36045056320401, 84.60575719649562]] )
accuracies_train.extend( [[91.84549356223177, 92.27467811158799, 89.69957081545064, 89.05579399141631], [92.91845493562232, 92.27467811158799, 89.91416309012875, 89.91416309012875], [94.63519313304721, 94.4206008583691, 91.63090128755364, 90.55793991416309], [95.70815450643777, 88.62660944206009, 89.69957081545064, 90.55793991416309], [94.63519313304721, 91.63090128755364, 90.55793991416309, 86.05150214592274]] )


data_test = np.array(accuracies)
data_train = np.array(accuracies_train)
data_test = make_data_boundaries(data_test)
data_train = make_data_boundaries(data_train)

plot(np.r_[ data_test, data_train[1:]], TITLE+'\nArgs: '+ARGS, 'proportion updated train data [%]', 'Balanced Accuracy [%]', ['test data', 'train data'], minmaxboundaries4plot=True)

N+=1
TITLE = 'Experiment '+str(N)+', (Right) 17% initial data 17% update (Seed independent)'
ARGS = '7 static gestures, all_defined, take_every_10, 1s, interpolate, 0.3split (tested on 30% data), 3000iter, 3000iupdate'
accuracies = [[ 17.5, 35, 52.5, 70 ]]
accuracies_train = [[ 17.5, 35, 52.5, 70 ]]
accuracies.extend( [[93.61702127659575, 94.49311639549437, 93.99249061326658, 93.11639549436796], [88.48560700876095, 86.85857321652065, 84.48060075093868, 80.72590738423028], [91.73967459324155, 91.98998748435544, 91.6145181476846, 93.99249061326658], [92.74092615769712, 88.36045056320401, 86.35794743429287, 88.6107634543179], [89.73717146433042, 89.98748435544431, 88.73591989987484, 86.98372966207761]] )
accuracies_train.extend( [[91.84549356223177, 92.27467811158799, 92.7038626609442, 90.77253218884121], [92.7038626609442, 90.34334763948499, 86.05150214592274, 83.04721030042919], [94.63519313304721, 94.63519313304721, 90.77253218884121, 95.06437768240343], [95.70815450643777, 87.55364806866953, 88.8412017167382, 91.84549356223177], [94.63519313304721, 91.41630901287554, 92.7038626609442, 90.55793991416309]] )

data_test = np.array(accuracies)
data_train = np.array(accuracies_train)
data_test = make_data_boundaries(data_test)
data_train = make_data_boundaries(data_train)

plot(np.r_[ data_test, data_train[1:]], TITLE+'\nArgs: '+ARGS, 'proportion updated train data [%]', 'Balanced Accuracy [%]', ['test data', 'train data'], minmaxboundaries4plot=True)


N+=1
TITLE = 'Experiment '+str(N)+', Take every 3 time-samples (Seed independent)'
ARGS = '7 static gestures, all_defined, take_every_3, 1s, interpolate, 0.25split (tested on 25% data), <1000-30000>iter'

accuracies =       [[1000, 3000, 4000, 5000, 6000, 10000, 20000, 30000]]
accuracies_train = [[ 1000, 3000, 4000, 5000, 6000, 10000, 20000, 30000 ]]
accuracies.append( [87.66368022053757, 85.9407305306685, 94.62439696760855, 95.4514128187457, 91.592005513439, 91.66092350103378, 72.29496898690559, 68.22880771881461] )
accuracies_train.append( [88.25287356321839, 84.75862068965517, 94.82758620689656, 93.81609195402298, 90.96551724137932, 93.05747126436782, 76.02298850574712, 69.47126436781609] )
accuracies.append([85.11371467953136, 90.90282563749138, 94.0730530668504, 85.25155065472089, 92.28118538938664, 90.6960716747071, 77.60165403170227, 68.22880771881461])
accuracies_train.append([84.73563218390805, 92.41379310344827, 93.24137931034483, 86.71264367816092, 93.19540229885057, 90.04597701149424, 78.06896551724138, 69.47126436781609])
accuracies.append( [83.3907649896623, 96.3473466574776, 91.86767746381805, 89.17987594762234, 88.42177808407995, 90.6960716747071, 77.60165403170227, 68.22880771881461] )
accuracies_train.append( [81.17241379310344, 96.20689655172414, 90.73563218390804, 88.62068965517241, 90.6896551724138, 90.04597701149424, 78.06896551724138, 69.47126436781609] )
accuracies.append( [78.49758787043419, 91.0406616126809, 93.86629910406616, 89.73121984838043, 90.6896551724138, 90.04597701149424, 78.06896551724138, 69.47126436781609] )
accuracies_train.append( [79.08045977011494, 90.20689655172414, 94.6896551724138, 89.74712643678161, 90.6896551724138, 90.04597701149424, 78.06896551724138, 69.47126436781609] )


data_test = np.array(accuracies)
data_train = np.array(accuracies_train)
data_test = make_data_boundaries(data_test)
data_train = make_data_boundaries(data_train)

plot(np.r_[ data_test, data_train[1:]], TITLE+'\nArgs: '+ARGS, 'iterations [-]', 'Balanced Accuracy [%]', ['test data', 'train data'], minmaxboundaries4plot=True)

N+=1
TITLE = 'Experiment '+str(N)+', Dataset proportion (from 10% to 90%) (no updates, only learning with given proportion) (Seed independent)'
ARGS = '7 static gestures, all_defined, take_every 10, 1s, interpolate, 3000iter (tested on rest of the train data)'
accuracies =       [[10, 20, 30, 40, 50, 60, 70, 80, 90]]
accuracies_train = [[10, 20, 30, 40, 50, 60, 70, 80, 90]]
accuracies.append( [94.90818030050083, 92.48826291079813, 85.568669527897, 91.11389236545682, 91.73553719008265, 92.86384976525821, 93.4918648310388, 94.55909943714822, 91.01123595505618] )
accuracies_train.append( [96.99248120300751, 92.4812030075188, 86.96741854636592, 92.10526315789474, 93.38842975206612, 92.86161552911709, 94.04186795491142, 94.36355096289337, 88.64300626304802] )
accuracies.append( [91.86143572621035, 91.22065727699531, 90.50429184549357, 91.36420525657071, 93.53869271224643, 93.33333333333333, 94.11764705882352, 93.05816135084429, 96.25468164794007] )
accuracies_train.append( [93.60902255639097, 94.3609022556391, 91.60401002506265, 92.19924812030075, 91.8106686701728, 93.17470256731372, 93.55877616747182, 92.86049788633161, 95.36534446764092] )
accuracies.append( [92.48747913188647, 93.9906103286385, 90.55793991416309, 85.23153942428036, 92.3365890308039, 92.95774647887323, 94.24280350438048, 90.0562851782364, 89.8876404494382] )
accuracies_train.append( [95.86466165413535, 95.67669172932331, 92.73182957393483, 85.71428571428571, 94.06461307287753, 93.36255479023168, 92.7536231884058, 89.33771723813997, 91.98329853862212] )
accuracies.append( [92.1118530884808, 94.27230046948357, 93.77682403433477, 94.68085106382979, 93.46356123215628, 95.58685446009389, 92.99123904881101, 91.74484052532833, 94.00749063670412] )
accuracies_train.append( [96.2406015037594, 96.42857142857143, 96.49122807017544, 95.0187969924812, 94.44027047332833, 95.36631183469004, 91.5727321524423, 93.89384687646782, 93.40292275574113] )
accuracies.append( [92.07011686143572, 89.48356807511738, 93.77682403433477, 95.61952440550688, 94.21487603305785, 94.74178403755869, 90.11264080100125, 92.12007504690432, 92.13483146067416] )
accuracies_train.append( [96.61654135338345, 90.6015037593985, 94.86215538847118, 94.83082706766918, 94.89105935386927, 94.30181590482154, 91.03596349973162, 92.29685298262095, 93.69519832985385] )


data_test = np.array(accuracies)
data_train = np.array(accuracies_train)
data_test = make_data_boundaries(data_test)
data_train = make_data_boundaries(data_train)

'The accuracy is not bigger, saturation already with 10%, with bigger dataset, there is a need to change some parameters (iterations?, .. ), test data should always be same!'
plot(np.r_[ data_test, data_train[1:]], TITLE+'\nArgs: '+ARGS, 'Proportion of train dataset [%]', 'Balanced Accuracy [%]', ['test data', 'train data'], minmaxboundaries4plot=True)

N+=1
TITLE = 'Experiment '+str(N)+', Dataset proportion (from 10% to 90%) (no updates, only learning with given proportion) (Seed independent)'
ARGS = '7 static gestures, all_defined, take_every 10, 1s, interpolate, [0.3,<0.1,0.9>]split (tested on 30% data), 3000iter'
accuracies =       [[10, 30, 50, 70, 90]]
accuracies_train = [[10, 30, 50, 70, 90]]
accuracies.append( [91.11389236545682, 94.36795994993741, 93.11639549436796, 93.4918648310388, 89.36170212765957] )
accuracies_train.append( [96.7741935483871, 96.23655913978494, 95.3813104189044, 93.48159509202453, 88.84248210023867] )
accuracies.append( [92.36545682102629, 92.24030037546935, 96.87108886107634, 95.36921151439299, 91.6145181476846] )
accuracies_train.append( [95.16129032258065, 95.16129032258065, 96.77765843179377, 95.93558282208589, 92.00477326968974] )
accuracies.append( [88.36045056320401, 93.99249061326658, 89.98748435544431, 93.11639549436796, 95.61952440550688] )
accuracies_train.append( [92.47311827956989, 94.98207885304659, 91.40708915145005, 93.3282208588957, 96.18138424821002] )
accuracies.append( [87.98498122653316, 93.11639549436796, 89.61201501877348, 91.48936170212765, 90.86357947434293] )
accuracies_train.append( [93.01075268817203, 90.86021505376344, 91.29967776584319, 91.4877300613497, 92.12410501193318] )
accuracies.append( [90.61326658322903, 93.99249061326658, 92.24030037546935, 96.12015018773467, 88.23529411764706] )
accuracies_train.append( [96.7741935483871, 94.44444444444444, 92.4812030075188, 97.00920245398773, 89.31980906921241] )


data_test = np.array(accuracies)
data_train = np.array(accuracies_train)
data_test = make_data_boundaries(data_test)
data_train = make_data_boundaries(data_train)

'training time from 7s to 40s, counts from [26, 25, 29, 30, 30, 15, 31] to [278, 243, 276, 258, 238, 188, 195]'
plot(np.r_[ data_test, data_train[1:]], TITLE+'\nArgs: '+ARGS, 'Proportion of train dataset [%]', 'Balanced Accuracy [%]', ['test data', 'train data'], minmaxboundaries4plot=True)

'''
#######################
First look conclusions:
#######################

At the start, I would like to say that, every trial or experiment can be easily reproduced, because every point has its given dataset and arguments, they are written for example in the plot. The assigned script `train.py` is attached to this plots.py.
Purpose: How accuracy will develop with increasing percents of dataset, these are not updates yet. What I wanted to see is increasing numbers of accuracy based on proportion of data. From the results, you can see, that training won't be efficient and train accuracy is collapsing based on the data.
When displaying development based on iterations, there is not much difference either. Bigger iterations are not better, system is saturated.

The important observation is difference between train and test data, which are different, so it the accuracy for test and accuracy for train data has no difference, then the model cannot learn better, for example with more iterations.

You can see that testing data are made from remaining dataset from whole dataset. From this plot it seems that from 20 recording samples above the difference is the same, but I got better results later.

In second section, I used continuous updates, I got nice increasing plot, but at the time I didn't knew that by assiging local_rv's to the model, I used the wrong variables.

Before I go the right variant I also looked more at the grid search. These results are correct in a way, but they have one major flaw, that they are as everything in probabilistic world, seed dependent. And I knew this would be the case, but then I found out that the results vary quite a lot for example about 4%. So when having different seed for randomness, I can have 91% accurate network but also 96% accurate network.

So I was finding some local maximums, but didn't consider the seed parameter. The question is that if I train the network and it will have 97% accuracy, if it will behave the same with other situations.

The I started doing seed independent trains. I used 5 different seeds and make average for every result I got, the result is this graph of low and high boundaries, which corresponds to one sigma standard deviation.

Right now you can see grid space search through number of iterations. Of course, the better way would be to use some stopping variable, to check if the loss is not stagnating and then overfitting is played here. But I cannot use control to loss function. Because the loss function is still decreasing even after the the local maximum in accuracy is observed, so I cannot use that. So the training network is still optimizing after the maximum is obtained, what improvements is it computing?

Another possible would be to check every x iterations to use MCMC and sample then drop the learning is accuracy failed. I know there is some feedback call, so I can try this.

The interpolation is maintaining the accuracy with bigger iterations.

These experiments are also wrong, there was problem with the variables. The first problem was that local_rv parameter does not do what I thaught it does. So I wanted to init the variable values from the scratch to the network. Then the problem which I had was that for a long time, the network seems that there is some problem, even with single iteration, model was not the same. The solutions was that saved weights were saved in columns, so I needed to transpose the weights matrix. When I did that, you can see that the model remained the same.

The main thing 5.8 computation time is very low, retraning with new data does not need minibatches, it will do the training given some iterations.

Other methods like FullRankADVI, SGD, OPVI
GPU acceleration
Minibatches were no better, training on full batch gave bigger accuracy.

What did I do wrong?
What to try?
Number of iterations pick based on dropping constraints?

Conclusion: Updates possible with new recordings. Especially, training new set of gestures with new data possible, easily get 90%, but will depend on what gestures it will be.
Make learning loop. Learning will be under 2 seconds is comfortably achieved, from recording use only valuable data. I will make script for putting the real data to the learning, not only sampling and it should be done.
'''




'Try online learning on real data. '

''



'thread for training online'
'see imported data from one learning'
'how many samples are there from 1s of recording'
''









N+=1
TITLE = 'Experiment '+str(N)+', 4 initial gestures, 3 new gestures update'
ARGS = '4+3 static gestures, all_defined, take_every 3, 1s, tested on 25% data'
accuracies =       [[2,      5,   50,  1000,  2500, 2505, 2510,  2520,  3500, 5000]]
accuracies =       [[4,      10,   50,  1000,  2500, 2505, 2510,  2520,  3500, 5000]]
accuracies.append( [40.4, 43.5, 65.9,  88.5,  94.4, 70.4, 94.5, 100.0, 100.0, 100.] )
accuracies.append( [91.3, 99.3, 99.9, 100.0, 100.0, 56.5, 58.4,  64.6, 70.5, 72.4] )


data_test = np.array(accuracies)
'UCB iteration is slower, the iteration itself ~200ms, but with sampling and accuracy check arround ~2s, where PyMC3 1000 iterations about 5 seconds, based on the dataset, \
UCB turn 100 in about 5 iterations in for initial learning and about 15 iterations for gesture updates, \
when I compare with my model, the model never reached 100%, \
my model has fewer n_hidden layers (50), their model has (1200), but when I set my model to 1200, with fewer iteartions, the ELBO is going down slowly and not as fast as UCB model '
plot(data_test, TITLE+'\nArgs: '+ARGS, 'Iterations [-]', 'Balanced Accuracy [%]', ['Uncertainty guided updates', 'Initial weights', 'Sampled from Posterior', 'Mean Sampled Posterior'])

accuracies =       [[4,      10, 10.1,   14,  20]]
accuracies.append( [91.3, 100.0, 0.0, 78.4, 98.5] )
accuracies.append( [88.5,  94.4, 0.0, 67.5, 72.4] )
accuracies.append( [78.125, 92.32954545454545, 0.0, 72.43947858472998,  88.08193668528864] )
accuracies.append( [75.85227272727273,  92.32954545454545, 0.0, 77.28119180633148,  85.84729981378027] )

data_test = np.array(accuracies)

'Recalculation to time, 1 UCB iteration to 500 (my model) iterations'
plot(data_test, TITLE+'\nArgs: '+ARGS, 'Time [s]', 'Balanced Accuracy [%]', ['Uncertainty guided updates', 'Initial weights', 'Proportion'])

N+=1
TITLE = 'Experiment '+str(N)+', 3 initial gestures, 2+2 new gestures update'
ARGS = '3+2+2 static gestures, all_defined, take_every 3, 1s, tested on 25% data'
accuracies =       [[4,      10, 10.1,   14,  20, 20.1, 24, 30]]
accuracies.append( [94.5, 100.0, 0.0, 78.4, 98.5, 0.0,  75.3, 100.0] ) # UCB
accuracies.append( [88.5, 93.4, 0.0, 67, 72.4, 0.0, 45.4, 48.5] ) # Initial weights
# Proportion 50 samples for every gesture (sampled from posterior)
accuracies.append( [81.29496402877699,  94.24460431654677, 0.0, 71.39479905437352, 91.96217494089835, 0.0, 81.56424581005587, 88.45437616387338] )
# Same as previous, but sampled mean 10 values
accuracies.append( [80.2158273381295,  94.60431654676259, 0.0, 73.75886524822694, 87.70685579196218, 0.0, 65.73556797020484, 83.42644320297951] )

data_test = np.array(accuracies)

'I think, accuracy visualized is better than Loss in my particular situation'
plot(data_test, TITLE+'\nArgs: '+ARGS, 'Time [s]', 'Balanced Accuracy [%]', ['Uncertainty guided updates', 'Initial weights', 'Sampled from Posterior', 'Mean Sampled Posterior'])


'''



'''


"""
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Invalid data & plots
Reason: Seed dependent data & other ..
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""

"""
Experiment 1:
    Legend: (<Train Data Proportion>/<Test Data Proportion>)
    Conclusion: Fig.1,2,3
Experiments 1.1 (Experiments.trainAdding10()):
Parameters:
    train.args = ['all_defined', 'take_every_10', '10000iter?']
Results:
    (10/90) --> 69% BA (only 72% on train?)
        - I can see that all gestures are learned well, only two are learned badly
    (20/80) --> 60% BA
    (30/70) --> 68% BA
    (40/60) --> 88% BA
    (50/50) --> 94% BA
    (60/40) --> 95.3% BA
    (70/30) --> 93.2% BA
    (80/20) --> 96.2% BA

Experiments 1.2 (Experiments.trainAdding10()):
Parameters:
    train.args = ['all_defined', 'take_every_10', '1s', 'interpolate', '3000iter', 'nsplit']
Results:
    (0.3/99.7) 36%/86% (8.3s) (Loss=977) ([1, 1, 1, 1, 1, 1, 1] record counts)
    (0.6/99.4) 64%/87% (4.4s) (Loss=1001) ([2, 2, 3, 2, 2, 2, 2] r.c.)
    (1/99) 73.7%/96% (4.4s) (Loss=1100) ([4, 4, 4, 4, 4, 3, 3] r.c.)
    (5/95) 88%/91% (7.5s) (Loss=1508) ([22, 20, 22, 19, 19, 16, 15] r.c.)
    (10/90) 92.3%/83.6% (14.7s) (Loss=1850)
    (10/90) 94.9%/96.9% (14.4s) (Loss=1788)
    (20/80) 94.4%/94.5% (15.6s) (Loss=2301)
    (20/80) 92.5%/92.5% (15.5s) (Loss=2328)
    (30/70) 85.5%/85.7% (22.3s) (Loss=2800)
    (40/60) 91.1%/92.4% (28.8s) (Loss=3267)
    (50/50) 90.9%/91.6% (36.2s) (Loss=3538)
    (60/40) 92.9%/92.9% (42.8s) (Loss=4022)
    (70/30) 93.5%/93.6% (50.6s) (Loss=4440)
    (80/20) 94.6%/94.1% (58.6s) (Loss=4756)
    (90/10) 89.5%/89.2% (65.3s) (Loss=5047)

Experiments 1.3 (Experiments.trainAdding10()):
Parameters:
    train.args = ['all_defined', 'take_every_10', '1s', 'interpolate', '3500iter', 'nsplit']
Results:
    (10/90) --> 95%/97.7% BA (test/train) (16.3s) (Loss=1562)
    (20/80) --> 92.67%/92.66% (test/train) (19s) (Loss=2092)
    (30/70) --> 87%/86% (26s) (Loss=2567)
    (40/60) --> 91,1%/92.2% (38s) (Loss=3013)
    (50/50) --> 91.8%/92.7% (44s) (Loss=3281)
    (60/40) --> 92.4%/92.5% (54s) (Loss=3785)
    (70/30) --> 92.1%/93% (61s) (Loss=4195)
    (80/20) --> 94.5%/94.3% (67s) (Loss=4498)
    (90/10) --> 90.3%/87.7% (75s) (Loss=4753)

Experiments 1.4 (Experiments.trainAdding10()):
Parameters:
    train.args = ['all_defined', 'take_every_10', '1s', 'interpolate', '4000iter', 'nsplit']
Results:
    (10/90) --> 94.4%/97.7% BA (18s) (Loss=1368)
    (20/80) --> 92.3%/92.7% BA (22.5s) (Loss=1892)
    (30/70) --> 86%/87% BA (32s)
    (40/60) --> 89.6%/91.8% (43s) (Loss=2800)
    (50/50) --> 91.6%/93.6% (51s) (Loss=3063)
    (60/40) --> 92.9%/91.2% (59s) (Loss=3567)
    (70/30) --> 91.5%/92.8% (66s) (Loss=3970)
    (80/20) --> 94.6%/94.2% (72.3s) (Loss=4264)
    (90/10) --> 88.8%/86.7% (86s) (Losss=4500)


"""

data_3000iter = np.array([[ 5. , 10. , 20. , 30. , 40. , 50. , 60. , 70. , 80. , 90. ],
       [88. , 93.6, 93.5, 85.5, 91.1, 90.9, 92.9, 93.5, 94.6, 89.5],
       [91. , 95.1, 93.5, 85.7, 92.4, 91.6, 92.9, 93.6, 94.1, 89.2]])
data_3500iter = np.array([[10.  , 20.  , 30.  , 40.  , 50.  , 60.  , 70.  , 80.  , 90.  ],
       [95.  , 92.67, 87.  , 91.1 , 91.8 , 92.4 , 92.1 , 94.5 , 90.3 ],
       [97.7 , 92.66, 86.  , 92.2 , 92.7 , 92.5 , 93.  , 94.3 , 87.7 ]])
data_4000iter = np.array([[10. , 20. , 30. , 40. , 50. , 60. , 70. , 80. , 90. ],
       [94.4, 92.3, 86. , 89.6, 91.6, 92.9, 91.5, 94.6, 88.8],
       [97.7, 92.7, 87. , 91.8, 93.6, 91.2, 92.8, 94.2, 86.7]])

# Fig.1
plot(data_3000iter, '1. (Fig.1) One-time learning each time, different proportion of train dataset used\n\
7 static gestures, args: all_defined, take_every_10, 3000 iter.', 'Amount train data from dataset [%]','Balanced Accuracy [%]', ['test data', 'train data'],\
'You can see network saturation only with 10% train data,\nafter that, accuracy decrease (to 86% BA, 30% train data)\nand then rise up ot 94% again with 80% train data.')

# Fig.2
data = np.array([[ 10. , 20. , 30. , 40. , 50. , 60. , 70. , 80. , 90. ],
                 [93.6, 93.5, 85.5, 91.1, 90.9, 92.9, 93.5, 94.6, 89.5],
                 [95.  , 92.67, 87.  , 91.1 , 91.8 , 92.4 , 92.1 , 94.5 , 90.3 ],
                 [94.4, 92.3, 86. , 89.6, 91.6, 92.9, 91.5, 94.6, 88.8]])
plot(data, '1. (Fig.2) One-time learning each time, different proportion of train dataset used\n7 static gestures, all_defined, take_every_10, iter=[3000, 3500, 4000]. Testing on remaining data', 'Amount train data from dataset [%]','Balanced Accuracy [%]', ['3000 iters', '3500 iters', '4000 iters'])


# Fig.3
data = np.array([#[0.3,0.6 ,1.  ,5. ,10.],
          [1. ,2.  ,4.  ,19.,38.],
          [36., 64.,73.7,88.,93.6],
          [86.,87., 96., 91., 96.9]
            ])
plot(data, '1. (Fig.3) One-time learning each time, different proportion of train dataset used\n7 static gestures, all_defined, take_every_10. Testing on remaining data', 'Number of records from each gesture [-]', 'Balanced Accuracy [%]', ['test data', 'train data'], 'Note: 38 recording samples for each gesture is ~10% train data')




"""
Experiment 2.1 (Experiments.trainAdding4updates()):
Parameters:
    train.args = ['all_defined', 'take_every_10']
Results:
    10% init. train, 90% test, 66% BA (Loss 969)
    10% init. train + 10% update train --> 73.5% BA (Loss 904)
    10% init. train + 2*10% update train --> 83.5% BA (Loss 847)
    10% init. train + 3*10% update train --> 71% BA (Loss 771)
    10% init. train + 4*10% update train --> 84% BA (Loss 735)
    10% init. train + 5*10% update train --> 78% BA (Loss 666)
    10% init. train + 6*10% update train --> 75% BA (Loss 620)
    10% init. train + 7*10% update train --> 80.1% BA (Loss 579)
    10% init. train + 8*10% update train --> 82.1% BA (Loss 554)
    10% init. train + 9*10% update train --> 79.5 % BA
Notes:
    - It does not learn to 100% even on train, reason?
Conclusion
    Not proper parameters, will try again with '1s' and 'interpolate'.

Experiment 2.2 (Experiments.trainAdding4updates()):
Parameters:
    train.args = ['all_defined', 'take_every_10']
Results:
    25% init. train, 81.1% BA (Loss 2273)
    25% init. train + 25% update train, 82.2% BA (Loss 3081)
    25% init. train + 2*25% update train, 86.8% BA (Loss 3050)
    25% init. train + 3*25% update train, 89.1% BA (Loss 3080)

---
Experiment 2.3 (Experiments.trainAdding4updates()):
Parameters:
    train.args = ['all_defined', 'take_every_10', '1s', 'interpolate', '3000iter', '0.3split', '50n_hidden']
Results:
    1rec./each gesture, init. train, 33.1%/100% BA (8.2s) (Loss 958)
    updated with 2rec./each gesture, 57.8%/86% BA (4.1s) (Loss 1045)
    updated with 5rec./each gesture, 78.8%/100% BA (4.6s) (Loss 1125)
    updated with 10rec./each gesture, 87.2%/85.6% BA (5.7s) (Loss 1266)
    updated with 20rec./each gesture, 90.3%/100% BA (7.4s) (Loss 1477)

---
Experiment 2.4 (Experiments.trainAdding4updates()):
Parameters:
    Arguments for training are:  ['all_defined', 'take_every_10', '1s', 'interpolate', '3000iter', '0.3split', '50n_hidden']
Results:
    (0.175%/30), 93.6%/92.7% BA (Loss 2283)
    (0.175+1*0.175u/30), 94.4%/92.0% BA (Loss 2257)
    (0.175+2*0.175u/30), 93.6%/94.0% BA (Loss 2238)
    (0.175+3*0.175u/30), 95.5%/94.6% BA (Loss 2187)

"""

TITLE = '2. (Fig.4) (Wrong) Continuous updates with 25% of train data'
ARGS = '7 static gestures, all_defined, take_every_10, 25% init and then 25% updates, testing on remaining data'
data = np.array([[ 25. ,  50. ,  75. , 100. ],
   [ 81.1,  82.2,  86.8,  89.1]])
plot(data, TITLE+'\nArgs: '+ARGS, 'Updating with 25% train data each time [%]', 'Balanced Accuracy [%]', ['test data'], 'Note: Accuracy didn\'t reached the maximum, because all arguments were not chosen properly' )

TITLE = '2. (Fig.5) (Wrong) Continuous updates with n records for every gesture'
ARGS = '7 static gestures, all_defined, take_every_10, 1s, interpolate, 3000iter, 0.3split, 50n_hidden, tested on remaining data'
data = np.array([[  1. ,   2. ,   5. ,  10. ,  20. ],
   [ 33.1,  57.8,  78.8,  87.2,  90.3],
   [100. ,  86. , 100. ,  85.6, 100. ]])
plot(data, TITLE+'\nArgs: '+ARGS, 'Update with n=[1,2,5,10,20] gestures', 'Balanced Accuracy [%]', ['test data', 'train data'])


TITLE = '2. (Fig.6) (Wrong) Continuous updates with 17.5% train data'
ARGS = '7 static gestures, all_defined, take_every_10, 1s, interpolate, 3000iter, 0.3split, 50n_hidden. 17.5% init and then 17.5% updates, tested on 30% test data'
data = np.array([[ 0.175,  0.35 ,  0.525,  0.7  ],
                  [92.36545682102629, 91.98998748435544, 91.23904881101377, 94.74342928660826],
                  [93.61702127659575, 94.11764705882352, 93.61702127659575, 95.36921151439299],
                  [90.36295369211514, 91.6145181476846, 91.98998748435544, 96.24530663329162],
                  ])

data4train = np.array([
[ 0.175,  0.35 ,  0.525,  0.7  ],
[92.7  , 92.   , 94.   , 94.6  ],
])

data = make_data_boundaries(data)
plot(data, TITLE+'\nArgs: '+ARGS, 'Updating with 17.5% train data each time [%]', 'Balanced Accuracy [%]', ['test data', 'train data'], rewrite_lower_boundaries=91, minmaxboundaries4plot=True)


"""
---
Experiment 3.1 (Experiments.trainAdding1000iter())
Parameters:
    train.args = ['all_defined', 'take_every_10', '0.25split']
Results:
    1000 iterations --> 70% BA (test)
    2000 iterations --> 87.9% BA (test)
    3000 iterations --> 89.7% BA (test)
    4000 iterations --> 89.0% BA (test)
    5000 iterations --> 88.8% BA (test)
    6000 iterations --> 83.5% BA (test)
    7000 iterations --> 81.9% BA (test)
    8000 iterations --> 86.3% BA (test)
    9000 iterations --> 86.0% BA (test)
    10000 iterations --> 89% BA (test)
    11000 iterations --> 88.7% BA (test)
    12000 iterations --> 83.6% BA (test)
Conclusion:
    Saturation on 3000 iterations
"""

TITLE = '(Fig. a) (Seed dependent) Searching for local maximum with variable of iterations'
ARGS = '7 static gestures, all_defined, take_every_10, 0.25split, 50n_hidden, tested on remaining data'
data = np.array([
    [1000, 70],
    [2000, 87.9],
    [3000, 89.7],
    [4000, 89.0],
    [5000, 88.8],
    [6000, 83.5],
    [7000, 81.9],
    [8000, 86.3],
    [9000, 86.0],
    [10000, 89],
    [11000, 88.7],
    [12000, 83.6]]).T
plot(data, TITLE+'\nArgs: '+ARGS, 'Update with iter=<1000,12000>', 'Balanced Accuracy [%]', ['test data', 'train data'])

"""
Experiment 3.2 (Experiments.trainAdding1000iter())
Parameters:
    train.args = ['all_defined', '<0.25-0.3>split', 'take_every', '1s', 'interpolate']
Results:
    iter=2000, 92.2% (477s) (Loss=24800)
    iter=2500, 93.5% (10min.) (Loss=22700)
    iter=50000, 88.2% (3h.) (Loss=4372)
"""
TITLE = '(Fig. b) (Seed dependent) Searching for local maximum with variable of iterations'
ARGS = '7 static gestures, all_defined, take_every, 1s, interpolate, 0.25split, 50n_hidden, tested on remaining data'
data = np.array([
    [2000, 92.2],
    [2500, 93.5],
    [50000, 88.2]]).T
plot(data, TITLE+'\nArgs: '+ARGS, 'Update with iter=<1000,12000>', 'Balanced Accuracy [%]', ['test data', 'train data'])

"""
Experiment 3.3 (Experiments.trainAdding1000iter())
Parameters:
    train.args = ['all_defined', '0.25split', 'take_every_10', '1s', 'interpolate']
Results:
    iter=1500, 94.8% BA (33s) (Loss=5700)
    iter=2000, 96.37% BA (41s) (Loss=5072)
    iter=2250, 96.5% BA (47s) (Loss=4868)
    iter=2500, 97.3% BA (50s) (Loss=4695)
    iter=3000, 97.37% BA (60.8s) (Loss=4393)
    iter=3500, 97.37% (69.5s) (Loss=4131)
    iter=4000, 97.0% BA (78.8s) (Loss=3895)
    iter=5000, 95.36% BA (97s)
    iter=7500, 93.1% (141s) (Loss=2851)
    iter=10000, 90% BA (188s)
    iter=15000, 85% BA (280s)
Conclusion:
    After 1500 iterations, model is usable (~30s). iter=3000s local maximum extreme
    After that, model gets worse, why? It is not the case of overfitting, because it would have got better the train dataset and test dataset would be worse, but it actually get worse the train dataset too.
"""
TITLE = '(Fig. c) Searching for local maximum with variable of iterations'
ARGS = '7 static gestures, all_defined, take_every_10, 1s, interpolate, 0.25split, seed=93457, 50n_hidden, tested on remaining data'
data = np.array([
    [1500, 94.8],
    [2000, 96.37],
    [2250, 96.5],
    [2500, 97.3],
    [3000, 97.37],
    [3500, 97.37],
    [4000, 97.0],
    [5000, 95.36],
    [7500, 93.1],
    [10000, 90],
    [15000, 85]
    ]).T
plot(data, TITLE+'\nArgs: '+ARGS, 'Update with iter=<1500,15000>', 'Balanced Accuracy [%]', ['test data', 'train data'])

"""
Experiment 3.4 (Experiments.trainAdding1000iter())
Parameters:
    train.args = ['all_defined', '0.25split', 'take_every_10', '1s']
Results:
    iter=2500, 88.6% BA (37.6s) (Loss=3900)
    iter=4000, 91.8% BA (56.7s) (Loss=3109)
    iter=4500, 92.7% BA (61s) (Loss=2922)
    iter=5000, 93.8% BA (66.8s) (Loss=2742)
    iter=6000, 93.67% BA (80s) (Loss=2462)
    iter=7000, 92.55% BA (92s) (Loss=2250)
    iter=10000, 90% BA (129s) (92.6% BA test)
Conclusion:
    Saturation on 93,8% BA if input data is not interpolated
"""
TITLE = '(Fig. d) Searching for local maximum with variable of iterations'
ARGS = '7 static gestures, all_defined, take_every_10, 1s, 0.25split, seed=93457, 50n_hidden, tested on remaining data'
data = np.array([
    [2500, 88.6],
    [4000, 91.8],
    [4500, 92.7],
    [5000, 93.8],
    [6000, 93.67],
    [7000, 92.55],
    [10000, 90]]).T
plot(data, TITLE+'\nArgs: '+ARGS, 'Update with iter=<2500,10000>', 'Balanced Accuracy [%]', ['test data', 'train data'])

"""
Experiment 3.5 (Experiments.trainAdding1000iter())
Parameters:
    train.args = ['all_defined', '0.25split', 'take_every_10']
Results:
    iter=10000, 75% BA (579s)
    iter=15000, (will see if any improvement) 74% BA (900s)
Conclusion:
    Cutting the recording to 1s is important, I don't see improvement in adding more iterations
"""
TITLE = '(Fig. e) Searching for local maximum with variable of iterations'
ARGS = '7 static gestures, all_defined, take_every_10, 0.25split, seed=93457, 50n_hidden, tested on remaining data'
data = np.array([
    [10000, 75],
    [15000, 74]
    ]).T
plot(data, TITLE+'\nArgs: '+ARGS, 'Update with iter=<10000,15000>', 'Balanced Accuracy [%]', ['test data', 'train data'])

"""
Experiment 4.1 (n_hidden & iter)
Parameters:
    train.args = ['all_defined', '0.25split', 'take_every_10', '1s', 'interpolate', ]
Results:
    n_hidden=[40] (3000iter), 92.7% BA (60s) (Loss=4207)
    n_hidden=[40] (4000iter), 91.7% BA (73s) (Loss=3783)
    n_hidden=[40] (5000iter), 90.2% BA (60s) (Loss=4207)
    n_hidden=[50] (3000iter), 97.13% BA (64s) (Loss=4561)
    n_hidden=[60] (3000iter), 90% BA (94% train) (73.1s) (Loss=4949)
    n_hidden=[60] (4000iter), 90.4% BA (94.2% train) (91s) (Loss=4384)
    n_hidden=[60] (5000iter), 90% BA (93% train) (105s) (Loss=3915)
Conclusion:
    Based on results, it seems that 50 nodes in hidden layer is optimal value for this kind of configuration.
"""

"""
Parameters:
    train.args = ['all_defined', '0.15split', 'take_every_10', '1s', 'interpolate', '4000iter', '50n_hidden']
Results:
    0.15split, 4000iter, 95.25% BA (92s) (Loss=4365)
    0.05split, 4000iter, 92-96.3% BA (97s) (Loss=4685)
    0.15split, 3500iter, TODO:
    0.05split, 3500iter, TODO:
    0.15split, 3000iter, TODO:
    0.05split, 3000iter, TODO:


Add-on: Possibility to compute
    1) F1 score: F1 = 2 (PPV x TPR)/(PPV+TPR)
    2) Matthews correlation coefficient (MCC)
    3) Hit rate TPR = TP/P, Precision PPV = TP/(TP+FP), Miss rate FNR = FN/P, Prevalence threshold PT, Threat score TS
    4) Accuracy (ACC) = (TP+TN)/(P+N)

Important realization:
    - I have never optimize the seed, beginner mistake was not to consider probability seed, then I found local maximum accuracy, but they were optimized for only one situation
"""


'''
Experiment 5.3, (Wrong) Update with 10% data (Seed independent)
Parameters:
    all_defined, take_every_10, 1s, interpolate, 0.25split, 10000samples, 4000 iter, tested on remaining data
'''
TITLE = 'Experiment 5.3, (Wrong) Update with 10% data (Seed independent)'
ARGS = '7 static gestures, all_defined, take_every_10, interpolate, 1s, 0.25split, 10000samples, 4000iter, tested on remaining data'

accuracies = [[    90,     100]]
accuracies_train = [[    90,    100]]
#Seed1
accuracies.append( [93.61702127659575, 90.73842302878599] )
accuracies_train.append( [93.67541766109785, 92.30310262529832] )
#Seed2
accuracies.append( [93.86733416770964, 93.2415519399249] )
accuracies_train.append( [95.34606205250597, 93.07875894988067] )
#Seed3
accuracies.append( [85.98247809762202, 92.24030037546935] )
accuracies_train.append( [87.76849642004774, 95.52505966587113] )
#Seed4
accuracies.append( [95.36921151439299, 88.48560700876095] )
accuracies_train.append( [96.18138424821002, 89.14081145584726] )
#Seed5
accuracies.append( [93.4918648310388, 91.98998748435544] )
accuracies_train.append( [93.67541766109785, 92.72076372315036] )


data_test = np.array(accuracies)
data_train = np.array(accuracies_train)
data_test = make_data_boundaries(data_test)
data_train = make_data_boundaries(data_train)


plot(np.r_[ data_test, data_train[1:]], TITLE+'\nArgs: '+ARGS, '90% vs 100% data [%]', 'Balanced Accuracy [%]', ['test data', 'train data'], minmaxboundaries4plot=True)



'''
Experiment 5.4, (Wrong) Update 3 times with 17% data (Seed independent)
Parameters:
    all_defined, take_every_10, 1s, interpolate, 0.3split, 3000iter, tested on 30% data
'''
TITLE = 'Experiment 5.4, (Wrong) Update 3 times with 17% data (Seed independent)'
ARGS = '7 static gestures, all_defined, take_every_10, interpolate, 1s, 0.3split, 3000iter, tested on 30% data'

accuracies = [[17, 35, 52, 70 ]]
accuracies_train = [[ 17, 35, 52, 70 ]]
#Seed1
accuracies.append([93.61702127659575, 94.11764705882352, 93.61702127659575, 95.36921151439299])
accuracies_train.append([91.84549356223177, 92.7038626609442, 93.56223175965665, 94.20600858369099])
#Seed2
accuracies.append([88.48560700876095, 92.61576971214018, 90.98873591989988, 91.36420525657071])
accuracies_train.append([92.7038626609442, 92.06008583690986, 87.55364806866953, 92.27467811158799])
#Seed3
accuracies.append([91.73967459324155, 93.2415519399249, 86.7334167709637, 92.74092615769712])
accuracies_train.append([94.63519313304721, 92.4892703862661, 89.48497854077253, 94.20600858369099])
#Seed4
accuracies.append([92.11514392991239, 94.86858573216522, 90.11264080100125, 90.98873591989988])
accuracies_train.append([95.70815450643777, 95.70815450643777, 90.34334763948499, 92.06008583690986])
#Seed5
accuracies.append([89.73717146433042, 93.86733416770964, 93.36670838548186, 89.23654568210263])
accuracies_train.append([94.63519313304721, 96.99570815450643, 95.70815450643777, 93.56223175965665])

data_test = np.array(accuracies)
data_train = np.array(accuracies_train)
data_test = make_data_boundaries(data_test)
data_train = make_data_boundaries(data_train)

plot(np.r_[ data_test, data_train[1:]], TITLE+'\nArgs: '+ARGS, 'proportion updated train data [%]', 'Balanced Accuracy [%]', ['test data', 'train data'], minmaxboundaries4plot=True)



TITLE = 'Experiment 5.4.1, (Wrong) Update 3 times with 17% data (Seed independent)'
ARGS = '7 static gestures, all_defined, take_every_10, interpolate, 1s, 0.3split, 3000iter, 1000iupdate, tested on 30% data'

accuracies = [[17, 35, 52, 70 ]]
accuracies_train = [[ 17, 35, 52, 70 ]]
accuracies.append( [93.99249061326658, 84.73091364205257, 78.5982478097622, 82.85356695869838] )
accuracies_train.append( [91.84549356223177, 80.04291845493562, 75.9656652360515, 81.11587982832617] )
accuracies.append( [88.48560700876095, 81.72715894868585, 74.59324155193993, 79.9749687108886] )
accuracies_train.append( [92.91845493562232, 81.54506437768241, 73.17596566523605, 83.90557939914163] )
accuracies.append( [91.73967459324155, 80.60075093867334, 74.71839799749687, 75.09386733416771] )
accuracies_train.append( [94.63519313304721, 81.54506437768241, 75.32188841201717, 78.54077253218884] )
accuracies.append( [92.74092615769712, 83.60450563204004, 77.09637046307884, 68.96120150187734] )
accuracies_train.append( [95.70815450643777, 84.54935622317596, 77.46781115879828, 68.88412017167383] )
accuracies.append( [89.73717146433042, 84.98122653316645, 72.84105131414267, 81.97747183979975] )
accuracies_train.append( [94.63519313304721, 87.55364806866953, 77.8969957081545, 85.62231759656652] )

accuracies2 = [[17, 35, 52, 70 ]]
accuracies2_train = [[ 17, 35, 52, 70 ]]
accuracies2.append( [93.61702127659575, 84.73091364205257, 78.3479349186483, 82.85356695869838] )
accuracies2_train.append( [91.84549356223177, 80.04291845493562, 76.1802575107296, 81.11587982832617] )
accuracies2.append( [88.48560700876095, 81.72715894868585, 74.59324155193993, 79.9749687108886] )
accuracies2_train.append( [92.7038626609442, 81.54506437768241, 73.17596566523605, 83.90557939914163] )
accuracies2.append( [91.73967459324155, 80.60075093867334, 72.71589486858574, 74.34292866082603] )
accuracies2_train.append( [94.63519313304721, 81.54506437768241, 74.46351931330472, 78.11158798283262] )
accuracies2.append( [92.11514392991239, 83.60450563204004, 77.09637046307884, 68.96120150187734] )
accuracies2_train.append( [95.70815450643777, 84.54935622317596, 77.46781115879828, 68.88412017167383] )
accuracies2.append( [90.11264080100125, 84.98122653316645, 72.84105131414267, 82.22778473091364] )
accuracies2_train.append( [93.56223175965665, 87.55364806866953, 77.8969957081545, 84.54935622317596] )

data_test = np.array(accuracies)
data_train = np.array(accuracies_train)
data_test2 = np.array(accuracies2)
data_train2 = np.array(accuracies2_train)
data_test = make_data_boundaries(data_test)
data_train = make_data_boundaries(data_train)
data_test2 = make_data_boundaries(data_test2)
data_train2 = make_data_boundaries(data_train2)

#np.r_[ data_test, data_train[1:], data_test2[1:], data_train2[1:] ]
plot(np.r_[ data_test, data_train[1:], data_test2[1:], data_train2[1:] ], TITLE+'\nArgs: '+ARGS, 'proportion updated train data [%]', 'Balanced Accuracy [%]', ['test data', 'train data'], minmaxboundaries4plot=True)






'''
Experiment 5.5, (Wrong) 90% trained, updated with 10% data, 50000 update iterations (Seed independent)
Parameters:
    all_defined, take_every_10, 1s, interpolate, 0.3split, 3000iter, tested on 30% data, 5iupdate
'''
TITLE = 'Experiment 5.5, (Wrong) 90% trained, updated with 10% data, 50000 update iterations (Seed independent)'
ARGS = '7 static gestures, all_defined, take_every_10, interpolate, 1s, 0.3split, 3000iter, tested on 30% data'

accuracies = [[90, 100 ]]
accuracies_train = [[ 90, 100 ]]
#Seed1
accuracies.append( [93.86733416770964, 92.61576971214018] )
accuracies_train.append( [93.31742243436753, 93.25775656324582] )
#Seed2
accuracies.append( [93.36670838548186, 89.11138923654568] )
accuracies_train.append( [94.98806682577565, 89.618138424821] )
#Seed3
accuracies.append( [87.35919899874844, 91.98998748435544] )
accuracies_train.append( [87.94749403341288, 92.12410501193318] )
#Seed4
accuracies.append( [94.74342928660826, 87.60951188986232] )
accuracies_train.append( [96.06205250596659, 86.87350835322196] )
#Seed5
accuracies.append( [93.4918648310388, 85.3566958698373] )
accuracies_train.append( [93.67541766109785, 82.51789976133651] )

data_test = np.array(accuracies)
data_train = np.array(accuracies_train)
data_test = make_data_boundaries(data_test)
data_train = make_data_boundaries(data_train)

plot(np.r_[ data_test, data_train[1:]], TITLE+'\nArgs: '+ARGS, 'proportion updated train data [%]', 'Balanced Accuracy [%]', ['test data', 'train data'], minmaxboundaries4plot=True)



'''
Experiment 5.7, (Wrong) Sample records for every gesture k=[1,2,5,10,20] (Seed independent)
Parameters:
    all_defined, take_every_10, 1s, interpolate, 0.3split, 3000iter, tested on 30% data, 1000iupdate
'''
TITLE = 'Experiment 5.7, (Wrong) Sample records for every gesture k=[1,2,5,10,20] (Seed independent)'
ARGS = '7 static gestures, all_defined, take_every_10, interpolate, 1s, 0.3split, 3000iter, tested on 30% data, 1000iupdate'

accuracies = [[1, 2, 5, 10, 20 ]]
accuracies_train = [[ 1, 2, 5, 10, 20 ]]

accuracies.extend([[34.79349186483104, 37.922403003754695, 51.18898623279099, 67.95994993742178, 74.09261576971214], [38.67334167709637, 16.64580725907384, 47.05882352941176, 58.072590738423024, 69.58698372966208], [55.819774718397994, 38.297872340425535, 57.82227784730913, 56.57071339173968, 74.34292866082603], [28.16020025031289, 33.917396745932415, 47.68460575719649, 55.819774718397994, 72.090112640801], [45.056320400500624, 26.282853566958696, 39.799749687108886, 62.57822277847309, 64.08010012515645]])
accuracies_train.extend([[85.71428571428571, 14.285714285714285, 57.14285714285714, 85.71428571428571, 85.71428571428571], [85.71428571428571, 42.857142857142854, 42.857142857142854, 71.42857142857143, 42.857142857142854], [71.42857142857143, 42.857142857142854, 57.14285714285714, 71.42857142857143, 71.42857142857143], [71.42857142857143, 57.14285714285714, 42.857142857142854, 57.14285714285714, 71.42857142857143], [100.0, 42.857142857142854, 57.14285714285714, 85.71428571428571, 71.42857142857143]])


data_test = np.array(accuracies)
data_train = np.array(accuracies_train)
data_test = make_data_boundaries(data_test)
data_train = make_data_boundaries(data_train)

plot(np.r_[ data_test, data_train[1:]], TITLE+'\nArgs: '+ARGS, 'k samples [-]', 'Balanced Accuracy [%]', ['test data', 'train data'], minmaxboundaries4plot=True)







#
