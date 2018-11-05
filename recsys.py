from surprise import Dataset
from surprise import Reader
from surprise import SVD, evaluate, print_perf, NMF, KNNBasic
import os


def main():
    pass
#    print('\nSVD\n')
#    q5()
#    print('\nPMF\n')
#    q6()
#    print('\nNMF\n')
#    q7()
#    print('\nUser Based CF\n')
#    q8()
#    print('\nItem Based CF\n')
#    q9()
    print('\nIBCFpearson\n')
    IBCFpearson()
    print('\nUBCFpearson\n')
    UBCFpearson()
    print('\nIBCFcosine\n')
    IBCFcosine()
    print('\nUBCFcosine\n')
    UBCFcosine()
    print('\nIBCFMSD\n')
    IBCFMSD()
    print('\nUBCFMSD\n')
    UBCFMSD()

    #for x in range(100):
    #   UBCFMSD(x)
    #   IBCFMSD(x)


def q5():
    file_path = os.path.expanduser('restaurant_ratings.txt')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path,reader=reader)

    data.split(n_folds=3)

    algo = SVD()
    perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)


def q6():
    file_path = os.path.expanduser('restaurant_ratings.txt')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path,reader=reader)

    data.split(n_folds=3)

    algo = SVD(biased=False)
    perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)


def q7():
    file_path = os.path.expanduser('restaurant_ratings.txt')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path,reader=reader)

    data.split(n_folds=3)

    algo = NMF()
    perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)


def q8():
    file_path = os.path.expanduser('restaurant_ratings.txt')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path,reader=reader)

    data.split(n_folds=3)

    algo = KNNBasic(sim_options={
        'user_based': True
    })
    perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)


def q9():
    file_path = os.path.expanduser('restaurant_ratings.txt')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path,reader=reader)

    data.split(n_folds=3)

    algo = KNNBasic(sim_options={
        'user_based': False
    })
    perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)


def IBCFpearson():
    file_path = os.path.expanduser('restaurant_ratings.txt')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path,reader=reader)

    data.split(n_folds=3)

    algo = KNNBasic(sim_options={
        'name': 'pearson',
        'user_based': False
    })
    perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)


def UBCFpearson():
    file_path = os.path.expanduser('restaurant_ratings.txt')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path,reader=reader)

    data.split(n_folds=3)

    algo = KNNBasic(sim_options={
        'name': 'pearson',
        'user_based': True
    })
    perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)


def IBCFMSD():
    file_path = os.path.expanduser('restaurant_ratings.txt')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path,reader=reader)

    data.split(n_folds=3)

    algo = KNNBasic(#k=x
        sim_options={
        'name': 'MSD',
        'user_based': False
    })
    perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)


def UBCFMSD():
    file_path = os.path.expanduser('restaurant_ratings.txt')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path,reader=reader)

    data.split(n_folds=3)

    algo = KNNBasic(#k=x,
        sim_options={
        'name': 'MSD',
        'user_based': True
    })
    perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)


def IBCFcosine():
    file_path = os.path.expanduser('restaurant_ratings.txt')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path,reader=reader)

    data.split(n_folds=3)

    algo = KNNBasic(sim_options={
        'name': 'cosine',
        'user_based': False
    })
    perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)


def UBCFcosine():
    file_path = os.path.expanduser('restaurant_ratings.txt')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path,reader=reader)

    data.split(n_folds=3)

    algo = KNNBasic(sim_options={
        'name': 'cosine',
        'user_based': True
    })
    perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)


if __name__ == '__main__':
    main()