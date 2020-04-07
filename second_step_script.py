import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
import sys
sys.path.insert(1, 'scripts/')
from gen_matrix import matrix_gen, get_ICA
from get_sample import get_sample, create_strings_for_dataset
from fft import fft_for_sample
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import operator



def generate_simple_dataset(size, pandas=False):
    h, w = size
    v = np.linspace(0.1, 40000, w)

    dataset = v
    for i in range(h-1):
        dataset = np.vstack((dataset, v))

    if pandas:
        return pd.DataFrame(dataset)

    return dataset


def func_for_1class(t, noise=0.5):
    return 2*np.cos(5*pi*t) + 5*np.cos(15*pi*t) + 3*np.cos(20*pi*t) + noise


def func_for_2class(t, noise=0.5):
    return 2*np.cos(5*pi*t) + 2*np.cos(10*pi*t) + 3*np.cos(20*pi*t) + noise


def func_for_3class(t, noise=0.5):
    return 2*np.cos(5*pi*t) + 10*np.cos(20*pi*t) + 3*np.cos(20*pi*t) + noise


def get_cosinus_matrix():
    data_simple = generate_simple_dataset((128, 20000))
    vec = data_simple[65]

    size = (128, 20000)
    class_ = size[1]//3
    class1 = [0,class_]
    class2 = [class_,class_*2]
    class3 = [class_*2, data_simple.shape[1]]

    vec[class1[0]:class1[1]] = func_for_1class(vec[class1[0]:class1[1]])
    vec[class2[0]:class2[1]] = func_for_2class(vec[class2[0]:class2[1]])
    vec[class3[0]:class3[1]] = func_for_3class(vec[class3[0]:class3[1]])

    data_simple[65] = vec

    return data_simple


def plot_fe_rf(rf, data_pca):
    tree_feature_importances = (rf.feature_importances_)
    sorted_idx = tree_feature_importances.argsort()

    y_ticks = np.arange(0, len(data_pca.columns))
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(15)
    ax.barh(y_ticks, tree_feature_importances[sorted_idx])
    ax.set_yticklabels(data_pca.columns[sorted_idx])
    ax.set_yticks(y_ticks)
    ax.set_title("Random Forest Feature Importances (MDI)")
    fig.tight_layout()
    plt.savefig('RF_PCA_Feature_Importances.png')
    plt.show()



def features_imp_pca(train_features, model_pca, X_pca, features_good, features_bad, features_normal):
    reward_best = 50
    reward_max = 10
    reward_med = 5
    reward_min = 1

    fe_imp = {}
    for feature in range(0, 2560):
        fe_imp['feature' + '_' + str(feature)] = 0

    for feature in tqdm(range(0, 2560)):
        reward = 0
        for component in range(0, 60):
            feature_value = model_pca.components_[component, feature]
            component_max = (pd.DataFrame(model_pca.components_).loc[component, :]).max()
            component_mean = (pd.DataFrame(model_pca.components_).loc[component, :]).mean()

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


def write_fe(sorted_d):
    with open('features_imp.txt', 'w') as fout:
        fout.write('name    weight')
        fout.write('\n')
        for i in sorted_d:
            fout.write(str(i[0]))
            fout.write(' ')
            fout.write(str(i[1]))
            fout.write('\n')


def table_recovery(train_features):
    ### Восстановим исходный вид таблицы, а именно 128x20x100 (102 в данном примере
    old_table = []
    for i in range(train_features.shape[0]):
        sample = pd.DataFrame(np.zeros((128, 20)))
        string = train_features.iloc[i, :]

        index_start = 0
        index_end = 128


        for s in range(20):
            sample.iloc[:, s] = string.iloc[index_start : index_end].values
            index_start = index_end
            index_end += 128

            if index_end > 2560:
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

def save_FE_items(FE_items):
    with open('FE_items_final_file.txt', 'w') as fout:
        fout.write('номер семпла, индекс, колонка (грубо говоря, адрес важной цифры)')
        fout.write('\n')
        fout.write('В семпле X в Y колонке, на Z индексе, какой-то важный сигнал и так далее')
        fout.write('\n')
        for i in FE_items:
            fout.write(str(i))
            fout.write('\n')







def main():
    size = (128, 20000)
    class_ = size[1]//3
    #Получаем новую матрицу смеси косинусов
    data = get_cosinus_matrix()
    ICA = get_ICA()
    #Перемножаем ICA и EEG матрицы
    matrix = np.matmul(ICA, data)
    #Разбиваем на матрицы классов, чтоб проще было делить на семплы
    matrix_class1 = matrix[:,0:class_]
    matrix_calss2 = matrix[:, class_:class_*2]
    matrix_calss3 = matrix[:, class_*2:data.shape[1]]
    #Получаем семплы для каждого класса
    sample_calss1 = get_sample(matrix_class1)
    sample_calss2 = get_sample(matrix_calss2)
    sample_calss3 = get_sample(matrix_calss3)
    #Преобразование Фурье
    samples_fft = list(map(abs, fft_for_sample(sample_calss1 + sample_calss2 + sample_calss3, first_n_elements=20)))

    sample_calss1_fft = samples_fft[:34]
    sample_calss2_fft = samples_fft[34:34*2]
    sample_calss3_fft = samples_fft[34*2:]

    #Создание строк для датасета, из матрицы 128*20 -> в вектор 2560
    sample_calss1_fft_str = create_strings_for_dataset(sample_calss1_fft)
    sample_calss2_fft_str = create_strings_for_dataset(sample_calss2_fft)
    sample_calss3_fft_str = create_strings_for_dataset(sample_calss3_fft)

    #Создание таблицы объекты-признаки
    print('class markup... ')

    #Класс 1
    data_class_1 = pd.DataFrame(data=np.zeros((34, 2560)))
    data_class_1['label'] = 1

    for i in tqdm(range(len(sample_calss1_fft_str))):
        data_class_1.loc[i, :-1] = sample_calss1_fft_str[i]


    #Класс 2
    data_class_2 = pd.DataFrame(data=np.zeros((34, 2560)))
    data_class_2['label'] = 2

    for i in tqdm(range(len(sample_calss2_fft_str))):
        data_class_2.loc[i, :-1] = sample_calss2_fft_str[i]


    #Класс 3
    data_class_3 = pd.DataFrame(data=np.zeros((34, 2560)))
    data_class_3['label'] = 3

    for i in tqdm(range(len(sample_calss3_fft_str))):
        data_class_3.loc[i, :-1] = sample_calss3_fft_str[i]


    data = pd.concat([data_class_1, data_class_2, data_class_3], axis=0)
    #print(data.shape) #(102, 2561)

    ## Понизим размерность до 60 компонент
    from sklearn.decomposition import PCA
    PCA = PCA(n_components=60)

    #Стандартизируем матрицу
    Scaler = StandardScaler()
    data_standart = Scaler.fit_transform((data).iloc[:, :-1])
    # Понижаем размерность
    data_pca = PCA.fit_transform(data_standart)

    data_pca = pd.DataFrame(data_pca)

    # data_pca['label'] = data['label'].values
    labels = data['label'].values

    #Обучим и посмотрим важные признаки с помощью RF
    rf = RandomForestClassifier()
    rf.fit(data_pca, labels)

    plot_fe_rf(rf, data_pca) ## save to current dir
    feature_importances = pd.DataFrame(rf.feature_importances_, index = data_pca.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
    #print(feature_importances.loc[:5, :])
    '''
    importance
    0	0.347334
    1	0.152703
    2	0.115552
    3	0.092159
    6	0.081289
    4	0.052368
    5	0.029165

    '''

    train_features = data_standart
    from sklearn.decomposition import PCA
    model = PCA(n_components=60).fit(train_features)
    X_pc = model.transform(train_features)

    features_good = ['0', '1', '2', '3', '6']
    features_normal = ['4','5','8','7']
    features_bad = list(map(str, np.arange(9, 60)))
    #получим исходные признаки с весами вклада в компоненты PCA
    print('get weight features...')
    print('~2min')
    d = features_imp_pca(train_features, model, X_pc, features_good, features_bad, features_normal)
    sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
    best_features = [sorted_d[i][0] for i in range(15)]
    #Запишем в файл
    write_fe(sorted_d)

    #print(best_features[:15])
    '''

    ('feature_1110', 175),
     ('feature_598', 174),
     ('feature_651', 171),
     ('feature_854', 171),
     ('feature_779', 169),
     ('feature_553', 168),
     ('feature_982', 167),
     ('feature_802', 165),
     ('feature_674', 163),
     ('feature_809', 163),
     ('feature_681', 159),
     ('feature_930', 159),
     ('feature_726', 157)

     '''

     #print(best_features[-15:])

    '''

     ('feature_68', -14),
     ('feature_92', -14),
     ('feature_115', -14),
     ('feature_18', -15),
     ('feature_48', -15),
     ('feature_8', -16),
     ('feature_81', -16),
     ('feature_118', -16),
     ('feature_12', -18),
     ('feature_38', -18),
     ('feature_58', -18),
     ('feature_9', -20),
     ('feature_61', -20),
     ('feature_63', -20),
     ('feature_127', -20)

     '''

     ### Главные признаки, с которыми будем рабоать, пометим специальной приставкой
    train_features = pd.DataFrame(train_features)
    for number_feature in list(best_features):
        number_feature = int(number_feature.split('_')[1])
        train_features.iloc[:, number_feature] = train_features.iloc[:, number_feature].apply(lambda x: str(x) + '_FE').values
    ### Восстановим исходный вид таблицы, а именно 128x20x100 (102 в данном примере)
    old_table = table_recovery(train_features)
    FE_items = search_important_features(old_table)
    save_FE_items(FE_items) #save to corrent dir


if __name__ == "__main__":
    main()
