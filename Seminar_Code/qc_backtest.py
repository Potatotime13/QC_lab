import numpy as np
from numpy.lib.shape_base import expand_dims
import pandas as pd
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px

def main():
    # Load Data
    additional_stocks = ['MSFT', 'GOOG', 'FB', 'AAPL','VOW.DE']
    stocks = ['ANZ.AX', 'AMC.AX', 'BHP.AX', 'BXB.AX', 'CBA.AX', 'CSL.AX', 'IAG.AX',]
    first_stock = 'AMP.AX'
    start_d = '2017-01-01'
    end_d = '2018-12-31'
    price = 'Adj Close'
    data = pd.DataFrame(yf.download(first_stock, start=start_d, end=end_d)[price])
    data.rename(columns={price:first_stock},inplace=True)
    risk = 0.5
    Amin = -1 * (len(stocks) + 1)
    Amax = len(stocks) + 2

    for stock in stocks:
        data[stock] = yf.download(stock, start=start_d, end=end_d)[price]

    data = data_cleaner(data)
    
    # get classic portfolio optimization solution
    data_mat = data.to_numpy()
    revenues = (data_mat[1:, :] / data_mat[:-1,:])-1
    revenues_mean = np.expand_dims(np.mean(revenues, axis=0), axis=0)
    variance = np.cov(revenues.T)

    weights = (1-risk)/(2*risk)*np.linalg.inv(variance) @ (revenues_mean.T)

    # create combinatorial solutions and compare to true minima
    results = []
    solutions = []
    for i in range(Amin,Amax):
        con_w = convert_min(weights.copy(), revenues_mean, variance, risk, i)
        min_, sol = evaluate(variance, revenues_mean, risk, i)
        print(con_w.T-sol.T)
        solutions.append(sol)
        results.append([min_[0,0], eval_weights(con_w, revenues_mean, variance, risk)[0,0]])
    
    # plot results
    results = np.array(results)
    rel_dist = 1 - (results[:,1]-results[:,0])/abs(results[:,0])
    plot = np.array([rel_dist,np.arange(Amin,Amax,1)])
    abs_min = eval_weights(weights, revenues_mean, variance, risk)[0,0]
    plt.subplots(figsize=(10,8))
    plt.bar(plot[1],plot[0])
    plt.ylabel('Qualität', fontsize=16)
    plt.xlabel('A', fontsize=16)
    plt.xticks(np.arange(Amin,Amax,1))
    plt.show()
    plt.subplots(figsize=(10,5))
    plt.scatter(np.arange(Amin,Amax,1),results[:,1], label='Approximierung')
    plt.scatter(np.arange(Amin,Amax,1),results[:,0], label='Minimum')
    plt.plot(np.arange(Amin,Amax,1),np.ones(results.shape[0])*abs_min)
    plt.ylabel('Kostenfunktion', fontsize=16)
    plt.xlabel('A', fontsize=16)
    plt.legend(fontsize=16)
    plt.show()
    

def convert_min(w,rev,var,risk,A):
    w = find_start(w,rev,var,risk)
    if A-sum(w) == 0:
        return w
    else:
        dir = (A-sum(w)) / abs(A-sum(w))
        dif = int(A-sum(w))
        for i in range(abs(dif)):
            grads = grad_weights(w + 0.0 * dir * (w != dir),rev,var,risk)
            grads += grad_weights(w + 0.5 * dir * (w != dir),rev,var,risk)
            grads += grad_weights(w + 1.0 * dir * (w != dir),rev,var,risk)
            grads = grads + (w == dir)
            order = np.argmin(abs(grads.T[0]))
            w[order] += dir
    return w

def find_start(w,rev,var,risk):
    act = [1,0,-1]
    order = np.argsort(-abs(w).T[0])
    for ind in order:
        start = 100
        for a in act:
            tmp = w.copy()
            tmp[ind,0] = a
            ev = eval_weights(tmp,rev,var,risk)
            if ev < start:
                best_a = a
                start = ev
        w[ind,0] = best_a
    return w

def convert_min2(w,rev,var,risk,A):
    if A >= 0:
        w = np.ones(w.size)
    else:
        w = -1 * np.ones(w.size)
    w = np.expand_dims(w,axis=1)
    if A-sum(w) == 0:
        return w
    else:
        dir = (A-sum(w)) / abs(A-sum(w))
        dif = int(A-sum(w))
        for i in range(abs(dif)):
            grads = grad_weights(w + 0.5 * dir * (w != dir),rev,var,risk)
            grads = grads + (w == dir)
            order = np.argmax(abs(grads.T[0]))
            w[order] += dir
    return w

def data_cleaner(df):
    for i in range(df.shape[1]):
        for j in range(df.shape[0]):
            if pd.isna(df.iloc[j,i]):
                if j > 0:
                    df.iloc[j,i] = df.iloc[j-1,i]
                else:
                    df.iloc[j,i] = df.iloc[j+1,i]
    return df

def create_plot_possibilities(df,A=4):
    plt.figure(figsize=(15,7))
    data = df - df.iloc[0,:]
    weights = get_combinations(df.shape[1],True,A)
    portfolio = data.to_numpy() @ weights.T
    x = np.arange(0,df.shape[0],1)
    for i in range(portfolio.shape[1]):
        plt.plot(x,portfolio[:,i])
    plt.title('Portfolio entwicklung aller Kombinationen', fontsize=16)
    plt.xlabel('Tage', fontsize=16)
    plt.ylabel('Totale Rendite', fontsize=16)
    plt.show()


def heatmap(data):
    data_mat = data.to_numpy()
    names = data.columns.tolist()

    revs = data_mat[1:,:] / data_mat[:-1,:] -1
    data.iloc[0,:] = np.zeros((1,data.shape[1]))
    data.iloc[1:,:] = revs
    data = data.corr().to_numpy().round(2)
    fig, ax = plt.subplots(figsize=(7,7))
    im = ax.imshow(data)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(names)):
        for j in range(len(names)):
            text = ax.text(j, i, data[i, j],
                        ha="center", va="center", color="w")

    ax.set_title("Korrelationen der Tagesrenditen")
    fig.tight_layout()
    plt.show()


def convert_weights(weights, grads, A):
    order = np.argsort(grads.T[0])
    conv = np.zeros(order.size)
    sum_pos = 0
    acts = np.array([-1,0,1])
    for i in range(order.size):
        dist = A-sum_pos
        posib = abs(dist - acts) <= (order.size - i -1)
        pos_act = acts[posib]
        wanted = weights[order[i],0]/abs(weights[order[i],0])
        if len(pos_act) == 1:
            conv[order[i]] = pos_act[0]
            sum_pos += conv[order[i]]
        elif np.isin(wanted,pos_act):
            conv[order[i]] = wanted
            sum_pos += conv[order[i]]
        else:
            conv[order[i]] = 0
            sum_pos += conv[order[i]]

    return conv

def eval_weights(w,rev,var,risk):
    return risk * w.T @ var @ w - (1-risk) * rev @ w

def grad_weights(w,rev,var,risk):
    #w = w * (abs(A)/sum(w))
    return 2 * risk * var @ w - (1-risk) * rev.T

def evaluate(var, rev, risk, A):
    weights = get_combinations(var.shape[0])
    min_val = 100
    for w in weights:
        w_t = np.array([w]).T
        if w_t.sum() == A:
            markov = risk * w_t.T @ var @ w_t - (1-risk) * rev @ w_t
            if markov < min_val:
                min_val = markov
                min_sol = w_t
    return min_val, min_sol

def ternary (n):
    if n == 0:
        return '0'
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(str(r))
    return ''.join(reversed(nums))

def get_combinations(num,check_A=False,A=4):
    control_obs = []

    if check_A:
        for i in range(3**num):
            tmp_n = [int(d) for d in ternary(i)]
            tmp_z = [0 for _ in range(num-len(tmp_n))]
            tmp = tmp_z + tmp_n
            tmp = (np.array(tmp)-1).tolist()
            if np.array(tmp).sum()==A:
                control_obs.append(tmp)
    else:
        for i in range(3**num):
            tmp_n = [int(d) for d in ternary(i)]
            tmp_z = [0 for _ in range(num-len(tmp_n))]
            tmp = tmp_z + tmp_n
            tmp = (np.array(tmp)-1).tolist()
            control_obs.append(tmp)

    return np.array(control_obs)    


if __name__ == '__main__':
    pass
