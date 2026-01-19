import numpy as np
import pandas as pd
import tqdm
from ._kernel import _binned_sigma_clip, _wlsm

def smooth_method(
    x:list, 
    y:list, 
    bins:list, 
    std_num:int=8, 
    std_const:float=1.8, 
    smooth_level:int=2, 
    iterate:int=5, 
    sample_max:int=600, 
    min_num:int=3, 
    kernel="discrete"
    ) -> pd.DataFrame:
    
    # first clean
    rawdata = _binned_sigma_clip(x=x, y=y, bins=bins, std_num=std_num, std_const=2.6, min_num=min_num)
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
                beta, sigma = _wlsm(data_slice.loc[mask, "x"], data_slice.loc[mask, "y"], data_slice.loc[mask, "weight"])
            else:
                beta, sigma = _wlsm(data_slice["x"], data_slice["y"], data_slice["weight"], iterate)
                
            # 標準差內數據保留
            y_pred = data["x"].copy()*beta[0]+beta[1]
            data["keep"] = np.abs(data["y"]-y_pred)/sigma<=std_const
            data_group.append(data)
            
    data_group = pd.concat(data_group, ignore_index=True)
    data_group = data_group.loc[:, ["x", "y", "tag", "keep"]]
    return data_group


def slope_method(
    x:list,
    y:list, 
    bins:list, 
    interval:float=0.75, 
    min_num:int=10, 
    weights:list=None, 
    interval_min_rate:float=0.8
    )->pd.DataFrame:
    
    if len(x)!=len(y):
        return "data length is inconsistaent"
    rawdata = pd.DataFrame({'x':x, 'y':y})
    for bin in bins:
        data_slice = rawdata[(rawdata["x"]>bin-interval) & (rawdata["x"]<bin+interval)]
        
        if len(data_slice)<min_num: 
            continue
            
        if data_slice["x"].max()-data_slice["x"].min()>data_slice:
            continue
        

        
        
        