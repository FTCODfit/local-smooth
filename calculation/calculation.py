"""
單檔測試
超導數據去躁、平均化模組
版本: Superconductor ED_3.2

功能摘要：
- 溫區分段的離群值抑制
- 加權回歸式平滑化與平均化
- 支援離散式平滑模式與權重策略調整

變更紀錄：
- 新增平滑化設定參數
- 新增離散式平滑化模式選擇
- 調整區間內資料點權重計算邏輯
"""

import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt

class Calcalation:
    def __init__(self, T_range:list=None):
        self.T_range = T_range # 標註
        
    def _binned_sigma_clip(self, x, y, bins, labels=None, std_num=8, std_const=1.8, min_num:int=3)->pd.DataFrame:
        rawdata = pd.DataFrame({'x':x, 'y':y})
        if len(x)!=len(y):
            return "data length is inconsistaent"
        rawdata['tag'] = pd.cut(rawdata['x'], bins=bins, labels=labels)
        x_all, y_all, tag_all = [], [], []
        tag_count = 0
        if not labels:
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

    def _wlsm(self, x, y, initial_weights=None, iterate=5, iterate_rate=0.7, iterate_stop=5e-3):
        if initial_weights is None:
            initial_weights =  np.ones(len(y))
        if len(x)!=len(y):
            raise ValueError("x and y length inconsistent")
        
        x, y, initial_weights = np.asarray(x), np.asarray(y), np.asarray(initial_weights)
        A, b = self._wls_1d(x, y, initial_weights)
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
                A, b = self._wls_1d(x, y, weights)
                beta = np.linalg.solve(A, b)
                y_pred = beta[0]*x+beta[1]
                residuals = y-y_pred
                sigma2 = weights*residuals**2
                sigma2 = np.clip(sigma2, 1e-12, None)
                z_score = np.abs((sigma2-np.mean(sigma2)) / np.std(sigma2)) # 計算離散分數
                
                if np.abs(beta_old[0]-beta[0])/np.abs(beta[0])+\
                    np.abs(beta_old[1]-beta[1])/np.abs(beta[1])<=2*iterate_stop:
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
    
    def smooth_method(self, x:list, y:list, bins:list, std_num:int=8, std_const:float=1.8, smooth_level:int=2\
        , iterate:int=5, sample_max:int=600, min_num:int=3, kernel="discrete") -> pd.DataFrame:
        # first clean
        rawdata = self._binned_sigma_clip(x=x, y=y, bins=bins, std_num=std_num, std_const=2.6, min_num=min_num)
        tags = rawdata['tag'].unique()
        
        data_group = []
        for tag in tqdm.tqdm(tags, desc="Smoothing Progress"):
            
            data_slice = rawdata[(rawdata['tag']>=tag-smooth_level) & (rawdata['tag']<=tag+smooth_level)].copy()
            # 防過少資料噪點
            if len(data_slice[data_slice["tag"]==tag])<min_num:
                continue
                       
            # 平滑化參數權重
            dist = np.abs(data_slice["tag"]-tag)
            if kernel.lower() == "discrete":
                data_slice["weight"] = 1/(dist+1)
            elif kernel.lower() == "gaussian":
                sigma = smooth_level/2
                data_slice["weight"] = np.exp(-(dist**2)/(2*sigma**2))
            
            if len(data_slice)>=std_num:
                data = data_slice.loc[data_slice["tag"]==tag].copy()
                
                # 篩選回歸直線的計算量控制
                if len(data_slice)>sample_max:
                    mask = data_slice.sample(n=sample_max, random_state=1).index
                    beta, sigma = self._wlsm(data_slice.loc[mask, "x"], data_slice.loc[mask, "y"], data_slice.loc[mask, "weight"])
                else:
                    beta, sigma = self._wlsm(data_slice["x"], data_slice["y"], data_slice["weight"], iterate)
                    
                # 標準差內數據保留
                y_pred = data["x"].copy()*beta[0]+beta[1]
                data["keep"] = np.abs(data["y"]-y_pred)/sigma<=std_const
                data_group.append(data)
                
        data_group = pd.concat(data_group, ignore_index=True)
        data_group = data_group.loc[:, ["x", "y", "tag", "keep"]]
        return data_group
                
                
    def _wls_1d(self, x, y, w):
        Sw = np.sum(w)
        Swx = np.sum(w*x)
        Swx2 = np.sum(w*x*x)
        A = np.array([[Swx2, Swx],
                      [Swx,  Sw]])
        Swy  = np.sum(w * y)
        Swxy = np.sum(w * x * y)
        b = np.array([Swxy, Swy])
        
        return A, b


if __name__=="__main__":
    
    np.random.seed(42)
    x = np.linspace(0, 100, 200)
    y = np.sin(x / 10) + np.random.normal(0, 0.5, size=len(x))
    outlier_idx = [20, 50, 120, 160]
    y[outlier_idx] += np.array([5, -6, 7, -5])

    bins = [0, 25, 50, 75, 100]
    bins = [i*10 for i in range(11)]
    calc = Calcalation()
    df_clean = calc.smooth_method(
        x=x,
        y=y,
        bins=bins,
        std_num=8,
        std_const=1.6,
        min_num=3
    )

    print(df_clean.head())
    print()
    print("原始點數:", len(x))
    print("清理後點數:", len(df_clean))
    print("去躁後點數:", len(df_clean[df_clean["keep"]]))
    print("tag 分佈:")
    print(df_clean["tag"].value_counts().sort_index())
    plt.figure()
    # plt.scatter(x, y, s=15)
    plt.scatter(df_clean["x"], df_clean["y"], s=15)
    plt.scatter(df_clean.loc[df_clean["keep"], "x"], df_clean.loc[df_clean["keep"], "y"], s=15)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Original data vs Binned Sigma-Clipped data")
    plt.show()
    print(df_clean)