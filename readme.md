# Local Smooth（局部穩健平滑工具）

本腳本主要用於 **實驗數據去躁與平均化** 的 Python 單檔工具，  
主要針對 **溫度或控制參數依賴的大量量測資料** 超導數據設計。

目前版本優先包裝一個核心功能()：  
**`smooth_method`：基於加權回歸的 Robust Local Smoothing**

---

## 設計動機

在實驗數據分析中，常見問題包括：

- 資料雜訊強烈、分佈不穩定  
- 局部區間內存在離群點（outliers）  
- 傳統平均或移動平均容易抹平物理結構  
- 全域擬合模型對局部異常極為敏感  

本工具的目標是提供一個：

> **在局部區間內估計趨勢，  
> 同時對離群點不敏感的平滑方法**

適合用於 **物理實驗、材料量測、溫度掃描資料** 等情境。

---

## 核心概念：Robust Local Smoothing

本方法的核心流程如下：

1. **x 軸分箱（Binned Segmentation）**  
   將資料依 x（如溫度）分段，建立局部結構。

2. **局部區間選取（Local Window）**  
   在每個中心 bin 附近，選取鄰近 bin 作為回歸資料。

3. **距離權重（Kernel Weighting）**  
   距離中心 bin 越近，權重越高（支援離散式與 Gaussian 權重）。

4. **加權線性回歸（Weighted Linear Regression）**

5. **IRLS 反覆加權（Robust Reweighting）**  
   依殘差大小動態降低離群點影響。

6. **局部標準差估計（Local σ Estimation）**

7. **僅對中心區間進行離群點判斷**

此設計主要提供：

- 在非預期曲線擬合下保留局部趨勢  
- 抑制離群值  
- 

---

### 目前功能

- 溫區分區的 sigma clipping  
- 加權局部線性回歸  
- IRLS 穩健權重更新  
- 局部標準差估計  
- 平滑後資料保留標記（`keep`）  
- 支援離散式與 Gaussian 權重策略  

### 預期加入

- 局部斜率／導數估計  
- 自適應平滑區間  
- 視覺化工具封裝 

---

## 使用方式


```python
from calcalation import Calcalation
import numpy as np

x = np.linspace(0, 100, 200)
y = np.sin(x / 10) + np.random.normal(0, 0.5, size=len(x))

bins = [i * 10 for i in range(11)]

calc = Calcalation()
df = calc.smooth_method(
    x=x,
    y=y,
    bins=bins,
    std_num=8,
    std_const=1.6,
    min_num=3
)
```
