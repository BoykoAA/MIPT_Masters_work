import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
import sys
sys.path.insert(1, './')
from gen_matrix import matrix_gen, get_ICA
from get_sample import get_sample, create_strings_for_dataset
from fft import fft_for_sample
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import collections
import operator
from sklearn import decomposition

## CONSTANTS

CHANALS = 128
N_COMPONENTS_PCA = 60
FREQ = 100
TIME_SEC = 200
NOISE = 0.5

TIME_SIZE_SEC = 3
STEP_TIME_SEC = 1

SAMPLE_SIZE = TIME_SIZE_SEC * FREQ
STEP_TIME = STEP_TIME_SEC * FREQ

LINSPACE = 0, TIME_SEC, FREQ*TIME_SEC




def generate_simple_dataset(linspace, chanals, pandas=False):
    p1,p2,p3 = linspace
    v = np.linspace(p1, p2, p3)

    dataset = v
    for i in range(chanals-1):
        dataset = np.vstack((dataset, v))

    if pandas:
        return pd.DataFrame(dataset)

    return dataset

def func_for_1class(t, noise=0.5):
    return 2*np.cos(5*2*pi*t) + 5*np.cos(15*2*pi*t) + 3*np.cos(20*2*pi*t) + np.random.normal(0,1)

def func_for_2class(t, noise=0.5):
    return 3*np.cos(5*2*pi*t) + 2*np.cos(15*2*pi*t) + 3*np.cos(20*2*pi*t) + np.random.normal(0,1)

def func_for_3class(t, noise=0.5):
    return 4*np.cos(5*2*pi*t) + 10*np.cos(15*2*pi*t) + 3*np.cos(20*2*pi*t) + np.random.normal(0,1)

def func_general(t, noise=0.5):
    return 9*np.cos(5*2*pi*t) + 10*np.cos(15*2*pi*t) + 3*np.cos(20*2*pi*t) + np.random.normal(0,1)

def get_cosinus_matrix(chanals, linspace):
    data_simple = generate_simple_dataset(linspace, chanals)
    vec = data_simple[0]

    size = (chanals, linspace[2])
    class_ = size[1]//3
    class1 = [0,class_]
    class2 = [class_,class_*2]
    class3 = [class_*2, data_simple.shape[1]]


    vec[class1[0]:class1[1]] = func_for_1class(vec[class1[0]:class1[1]])
    vec[class2[0]:class2[1]] = func_for_2class(vec[class2[0]:class2[1]])
    vec[class3[0]:class3[1]] = func_for_3class(vec[class3[0]:class3[1]])


    data_simple = func_general(data_simple)
    data_simple[0] = data_simple[1]
    # data_simple[10] = vec #########!!!!!!!!!!!!!!!!
    data_simple[100] = vec #########!!!!!!!!!!!!!!!!
    # data_simple[65] = vec

    size = data_simple.shape
    class_ = size[1] //3

    return data_simple, size, class_


def get_i_(sample_calss1):
    i_ = 0
    for i in range(len(sample_calss1)):
        for j in range(sample_calss1[0].shape[0]):
            if sample_calss1[i][j].shape[0] != SAMPLE_SIZE:
                if i_ == 0:
                    i_ = i
    return i_


def scoring_fi(feature_importances):
    above_zero = feature_importances['importance'][:np.sum(feature_importances['importance'] > 0)]
    mean_value = above_zero.mean()
    features_good = above_zero[above_zero > mean_value].index.tolist()
    features_normal = above_zero[above_zero <= mean_value].index.tolist()
    features_bad = [i for i in feature_importances.index.tolist() if i not in features_good and
                    i not in features_normal]


    features_good = [i for i in features_good if i not in ['[', ']', ',']]
    features_normal = [i for i in features_normal if i not in ['[', ']', ',']]
    features_bad = [i for i in features_bad if i not in ['[', ']', ',']]

    features_good = list(map(lambda x: str(x), features_good))
    features_normal = list(map(lambda x: str(x), features_normal))
    features_bad = list(map(lambda x: str(x), features_bad))


    return features_good, features_normal, features_bad

def rf_fit(data_pca, labels):
    fg = []
    fn = []
    fb = []
    for _ in range(200):
        rf = RandomForestClassifier()
        rf.fit(data_pca, labels)
        feature_importances = pd.DataFrame(rf.feature_importances_,
                                       index = data_pca.columns,
                                        columns=['importance']).sort_values('importance',ascending=False)

        features_good, features_normal, features_bad = scoring_fi(feature_importances)
        fg.extend(features_good)
        fn.extend(features_normal)
        fb.extend(features_bad)

    features_good = list(map(lambda x: x[0],
                               sorted(collections.Counter(fg).items(), key=operator.itemgetter(1), reverse=True)[:10]))
    features_normal = list(map(lambda x: x[0],
                               sorted(collections.Counter(fn).items(), key=operator.itemgetter(1), reverse=True)))

    features_normal = list(set(features_normal) - set(features_good))

    features_bad = list(set(fb) - set(features_good) - set(features_normal))

    return features_good, features_normal, features_bad


def features_imp_pca(train_features, model_pca, X_pca, features_good, features_bad, features_normal, size, FIRST_N_FFT):

    global N_COMPONENTS_PCA

    reward_best = 50
    reward_max = 10
    reward_med = 5
    reward_min = 1

    fe_imp = {}
    for feature in range(0, size[1]):
        fe_imp['feature' + '_' + str(feature)] = 0

    component_max_list = [abs(pd.DataFrame(model_pca.components_).loc[i, :]).max() for i in range(N_COMPONENTS_PCA)]
    component_mean_list = [abs(pd.DataFrame(model_pca.components_).loc[i, :]).mean() for i in range(N_COMPONENTS_PCA)]

    for feature in tqdm(range(0, size[0]*FIRST_N_FFT)):
        reward = 0
        for component in range(0, N_COMPONENTS_PCA):
            feature_value =abs( model_pca.components_[component, feature])
            component_max = component_max_list[component]
            component_mean = component_mean_list[component]

            comparison_max = component_max - component_max / 10
            comparison_med = component_max - component_max / 20
            comparison_min = component_mean



            if feature_value >= comparison_min:
                if str(component) in features_bad:
                    reward -= reward_min
                elif str(feature) in features_good or str(feature) in features_normal:
                    reward += reward_min

            if feature_value >= comparison_med:
                if str(component) in features_bad:
                    reward -= reward_med
                elif str(component) in features_normal:
                    reward += reward_med
                elif str(component) in features_good:
                    reward += reward_max

            if feature_value >= comparison_max:
                if str(component) in features_bad:
                    reward -= reward_max
                elif str(component) in features_normal:
                    reward += reward_max
                elif str(component) in features_good:
                    reward += reward_best #best

            if feature_value <= comparison_min:
                if str(component) in features_bad:
                    reward += reward_min
                elif str(component) in features_good or str(feature) in features_normal:
                    reward -= reward_min


        fe_imp['feature' + '_' + str(feature)] = reward

    return fe_imp


def table_recovery(train_features, FIRST_N_FFT, size):
    global N_COMPONENTS_PCA, CHANALS
    ### Восстановим исходный вид таблицы, а именно 128x20x100 (102 в данном примере
    old_table = []
    for i in tqdm(range(train_features.shape[0])):
        sample = pd.DataFrame(np.zeros((CHANALS, FIRST_N_FFT)))
        string = train_features.iloc[i, :]

        index_start = 0
        index_end = size[0]


        for s in range(FIRST_N_FFT):
            sample.iloc[:, s] = string.iloc[index_start : index_end].values
            index_start = index_end
            index_end += CHANALS

            if index_end > size[0]*FIRST_N_FFT:
                break

        old_table.append(sample.values)

    return old_table


def search_important_features(old_table):
    FE_items = []

    for table_number, table in tqdm(enumerate(old_table)):
        for column in range(table.shape[1]):
            for idx in range(table.shape[0]):
                if len(str(table[idx, column]).split('_')) > 1:
                    FE_items.append((table_number, idx, column))

    return FE_items



def run():
    print('Start')
    print('Make data')
    matrix, size, class_ = get_cosinus_matrix(chanals=CHANALS, linspace=LINSPACE)

    FastICA = decomposition.FastICA(n_components=CHANALS).fit(matrix.T)
    ICA = FastICA.transform(matrix.T)
    matrix = ICA.T

    matrix_class1 = matrix[:,0:class_]
    matrix_calss2 = matrix[:, class_:class_*2]
    matrix_calss3 = matrix[:, class_*2:matrix.shape[1]]
    #Получаем семплы для каждого класса
    print('Get samples')
    sample_calss1 = get_sample(matrix_class1, sample_size=SAMPLE_SIZE, step=STEP_TIME)
    sample_calss2 = get_sample(matrix_calss2, sample_size=SAMPLE_SIZE, step=STEP_TIME)
    sample_calss3 = get_sample(matrix_calss3, sample_size=SAMPLE_SIZE, step=STEP_TIME)
    i_ = get_i_(sample_calss1)
    sample_calss1 = sample_calss1[:i_]
    sample_calss2 = sample_calss2[:i_]
    sample_calss3 = sample_calss3[:i_]
    print('Fourier transform')
    samples_fft = list(fft_for_sample(sample_calss1 + sample_calss2 + sample_calss3, freq=FREQ))
    len_class = len(sample_calss1)
    sample_calss1_fft = samples_fft[:len_class]
    sample_calss2_fft = samples_fft[len_class:len_class*2]
    sample_calss3_fft = samples_fft[len_class*2:]
    FIRST_N_FFT = len(sample_calss1_fft[0][0])

    #Создание строк для датасета, из матрицы CHANNELS*FIRST_N_FFT -> в вектор
    sample_calss1_fft_str = create_strings_for_dataset(sample_calss1_fft)
    sample_calss2_fft_str = create_strings_for_dataset(sample_calss2_fft)
    sample_calss3_fft_str = create_strings_for_dataset(sample_calss3_fft)

    #Создание таблицы объекты-признаки

    #Класс 1
    data_class_1 = pd.DataFrame(data=np.zeros((len_class, size[0] * FIRST_N_FFT)))
    data_class_1['label'] = 1

    data_class_1 = np.array(data_class_1)

    for i in tqdm(range(len(sample_calss1_fft_str))):
        data_class_1[i, :-1] = sample_calss1_fft_str[i]


    #Класс 2
    data_class_2 = pd.DataFrame(data=np.zeros((len_class, size[0] * FIRST_N_FFT)))
    data_class_2['label'] = 2

    data_class_2 = np.array(data_class_2)

    for i in tqdm(range(len(sample_calss2_fft_str))):
        data_class_2[i, :-1] = sample_calss2_fft_str[i]


    #Класс 3
    data_class_3 = pd.DataFrame(data=np.zeros((len_class, size[0] * FIRST_N_FFT)))
    data_class_3['label'] = 3

    data_class_3 = np.array(data_class_3)

    for i in tqdm(range(len(sample_calss3_fft_str))):
        data_class_3[i, :-1] = sample_calss3_fft_str[i]

    data = np.vstack([data_class_1, data_class_2, data_class_3])
    data = pd.DataFrame(data)
    data.columns = [*data.columns[:-1], 'label']
    print(data.shape)


    ## Понизим размерность до 60 компонент
    from sklearn.decomposition import PCA
    PCA = PCA(n_components=N_COMPONENTS_PCA, random_state=100)
    data_standart = (data).iloc[:, :-1]
    # Понижаем размерность
    data_pca = PCA.fit_transform(data_standart)
    data_pca = pd.DataFrame(data_pca)
    labels = data['label'].values

    features_good, features_normal, features_bad = rf_fit(data_pca, labels)

    train_features = data_standart
    from sklearn.decomposition import PCA
    model = PCA(n_components=N_COMPONENTS_PCA, random_state=100).fit(train_features)
    X_pc = model.transform(train_features)
    d = features_imp_pca((train_features), model, X_pc, features_good, features_bad, features_normal, size, FIRST_N_FFT)
    sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
    best_features = [sorted_d[i][0] for i in range(10)]

    ### Главные признаки, с которыми будем рабоать
    train_features = pd.DataFrame(train_features)
    for number_feature in list(best_features):
        number_feature = int(number_feature.split('_')[1])
        train_features.iloc[:, number_feature] = train_features.iloc[:, number_feature].apply(lambda x: str(x) +
                                                                                              '_FE').values

    old_table = table_recovery(train_features, FIRST_N_FFT, size)
    FE_items = search_important_features(old_table)

    best_feat = list(map(lambda x: x[0],
    sorted(collections.Counter(list(map(lambda x: x[1], FE_items))).items(),
           key=operator.itemgetter(1), reverse=True)))

    ch = []
    for i in best_feat:
        ch.append(np.argmax(np.abs(FastICA.mixing_[:, i])))

    print(ch)
    return ch

run()
