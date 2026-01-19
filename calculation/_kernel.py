import pandas as pd
import numpy as np

def _binned_sigma_clip(x, y, bins, labels=None, std_num=8, std_const=1.8, min_num:int=3)->pd.DataFrame:
        rawdata = pd.DataFrame({'x':x, 'y':y})
        if len(x)!=len(y):
            return "data length is inconsistaent"
        rawdata['tag'] = pd.cut(rawdata['x'], bins=bins, labels=labels)
        x_all, y_all, tag_all = [], [], []
        tag_count = 0
        if labels is None:
            labels=bins[:-1] 
        for tag, data in rawdata.groupby(rawdata['tag'], observed=True):
            x1, y1 = np.asarray(data['x']), np.asarray(data['y'])
            if len(x1)<=min_num: 
                continue
            
            for _ in range(3):
                if len(x1)>=std_num:
                    y1_mean, y1_std = np.mean(y1), np.std(y1, ddof=1)
                    allow = np.abs(y1-y1_mean)/y1_std < std_const
                    x2, y2= x1[allow],  y1[allow]
                    if np.std(y2, ddof=1)<=0.85*y1_std:
                        x1, y1 = x2, y2
                        break
                    x1, y1 = x2, y2
                else: break
            tag_count+=1
            x_all.append(x1), y_all.append(y1), tag_all.append(np.full(len(y1), tag_count))
    
        x_all, y_all, tag_all = np.concatenate(x_all), np.concatenate(y_all), np.concatenate(tag_all)
        return pd.DataFrame({'x':x_all, 'y':y_all, 'tag':tag_all})
    
def _wlsm(x:list, y:list, initial_weights=None, iterate=5, iterate_rate=0.7, iterate_stop=5e-3, eps=1e-12):
    if initial_weights is None:
        initial_weights =  np.ones(len(y))
    if len(x)!=len(y):
        raise ValueError("x and y length inconsistent")
    
    x, y, initial_weights = np.asarray(x), np.asarray(y), np.asarray(initial_weights)
    A, b = _wls_1d(x, y, initial_weights)
    beta = np.linalg.solve(A, b)
    y_pred = beta[0]*x+beta[1]
    residuals = y-y_pred
    sigma2 = initial_weights*residuals**2 # 2 degree
    sigma2 = np.clip(sigma2, 1e-12, None)
    z_score = np.abs((sigma2-np.mean(sigma2)) / np.std(sigma2)) # 計算離散分數
    weights = initial_weights
    if iterate<1:
        pass
    else:
        for _ in range(iterate):
            beta_old = np.copy(beta)
            weights_old = np.copy(weights)
            weights = (np.exp(-z_score))*initial_weights
            weights =  iterate_rate*weights + (1-iterate_rate)*weights_old # 平滑迭代
            weights = weights/np.max(weights) # 歸一化
            weights = np.clip(weights, 1e-3, 10) # 限制最小權重避免趨於 0
            A, b = _wls_1d(x, y, weights)
            beta = np.linalg.solve(A, b)
            y_pred = beta[0]*x+beta[1]
            residuals = y-y_pred
            sigma2 = weights*residuals**2
            sigma2 = np.clip(sigma2, 1e-12, None)
            z_score = np.abs((sigma2-np.mean(sigma2)) / np.std(sigma2)) # 計算離散分數
            
            
            if np.abs(beta_old[0]-beta[0])/(np.abs(beta[0])+eps)+\
                np.abs(beta_old[1]-beta[1])/(np.abs(beta[1])+eps)<=2*iterate_stop:
                break
    # w_sum = np.sum(weights)
    # mx = np.sum(weights(x-np.mean(x)))/w_sum
    # my = np.sum(weights(x-np.mean(y)))/w_sum  
    # cov_xy = np.sum(weights*(x-mx)*(y-my))/w_sum
    # var_x  = np.sum(weights*(x-mx)**2)/w_sum
    # var_y  = np.sum(weights*(y-my)**2)/w_sum
    # r_w = cov_xy/np.sqrt(var_x*var_y) # 加權相關係數
    sigma = np.sqrt(np.sum(sigma2)/np.sum(weights))
    
    return beta, sigma # [slope, intercept], standard

def _wls_1d(x, y, w):
    Sw = np.sum(w)
    Swx = np.sum(w*x)
    Swx2 = np.sum(w*x*x)
    A = np.array([[Swx2, Swx],
                    [Swx,  Sw]])
    Swy  = np.sum(w * y)
    Swxy = np.sum(w * x * y)
    b = np.array([Swxy, Swy])
    
    return A, b