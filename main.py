import numpy as np
import kmeans
import common
import naive_em
import em

## For ProGit practice uncomment second two lines - merge conflict

# X = np.loadtxt("toy_data.txt")


# K-means
#
cost_array = np.zeros([5,4])
cost_array.fill(np.nan) # allocate non array
#
# for seed in range(5):
#
#     for K in range(1,5):
#
#         [mixture, post] = common.init(X,K,seed)
#         [mixture, post, cost] = kmeans.run(X, mixture, post)
#
#         cost_array[seed, K-1] = cost
#         cost_min = np.min(cost_array, axis=0) # min for each column
#
#         common.plot(X, mixture, post, 'K means  clustering')
#
# print("K means cost_arary", cost_array)
# print("K means cost min", cost_min)
#
#
#
# # 3. Expectation–maximization algorithm
#
#
# L_array = np.zeros([5,4])
# L_array.fill(np.nan) # allocate non array
#
#
# for seed in range(5):
#
#     for K in range(1,5):
#
#         [mixture, post] = common.init(X,K,seed)
#         [mixture, post, L] = naive_em.run(X, mixture, post)
#
#         L_array[seed, K-1] = L
#         L_min = np.min(L_array, axis=0) # min for each column
#
#         common.plot(X, mixture, post, 'EM clustering')
#
# print("EM L_array", L_array)
# print("EM L_min", L_min)
#
#
#
# # 5. Bayesian Information Criterion
#
# BIC_array = []
# seed = 0
#
# for K in range(1,5):
#
#     [mixture, post] = common.init(X,K,seed)
#     [mixture, post, L] = naive_em.run(X, mixture, post)
#
#     BIC = common.bic(X, mixture, L)
#
#     BIC_array.append(BIC)
#
#
# BIC_best = max(BIC_array) # least penalty BIC
# K_best = BIC_array.index(max(BIC_array))+1 # nest K index
#
# print("Best BIC", BIC_best)
# print("Best K", K_best)



# # 7. EM with delta
#
#
# L_array = np.zeros([5,4])
# L_array.fill(np.nan) # allocate non array
#
#
# for seed in range(5):
#
#     for K in range(1,5):
#
#         [mixture, post] = common.init(X,K,seed)
#         [mixture, post, L] = em.run(X, mixture, post)
#
#         L_array[seed, K-1] = L
#         L_min = np.min(L_array, axis=0) # min for each column
#
#         common.plot(X, mixture, post, 'EM clustering')
#
# print("EM L_array", L_array)
# print("EM L_min", L_min)

#

# 8. EM with delta for netflix_incomplete

X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt("netflix_complete.txt")

n, d = X.shape

K = 4
seed = 0

mix_conv, post_conv, log_lh_conv = em.run(X, *common.init(X, K, seed))

X_predict = em.fill_matrix(X, mix_conv)

rmse = common.rmse(X_gold, X_predict)

X = np.loadtxt("netflix_incomplete.txt")
K = [1, 12]

log_lh = [0, 0, 0, 0, 0]

best_seed = [0, 0]

mixtures = [0, 0, 0, 0, 0]

posts = [0, 0, 0, 0, 0]

rmse = [0., 0.]

for k in range(len(K)):
    for i in range(5):
        mixtures[i], posts[i], log_lh[i] = \
        em.run(X, *common.init(X, K[k], i))

    print("Clusters ", K[k])
    print("highest log_lh ", np.max(log_lh))

    best_seed[k] = np.argmax(log_lh)
    X_predict = em.fill_matrix(X, mixtures[best_seed[k]])
    print("rmse=", common.rmse(X_gold, X_predict))














