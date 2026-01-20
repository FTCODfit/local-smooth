# Local Smooth（局部穩健平滑模組）

本腳本主要用於 **實驗數據去躁與平均化** 的 Python 模組工具，  
主要針對 **溫度或控制參數依賴的大量量測資料** 超導數據設計。

---

## 設計動機

在實驗數據分析中，常見問題包括：

- 資料雜訊強烈、分佈不穩定  
- 局部區間內存在離群點（outliers）  
- 傳統平均或移動平均容易抹平物理結構  
- 全域擬合模型對局部異常極為敏感  

本模組的目標是提供一個：

> **在局部區間內估計趨勢，  
> 同時對離群點不敏感的平滑方法**

適合用於 **物理實驗、材料量測、溫度掃描資料** 等大量原始數據。

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


---

### 目前功能

- 溫區分區的 sigma clipping  
- 加權局部線性回歸  
- IRLS 穩健權重更新  
- 局部標準差估計  
- 平滑後資料保留標記（`keep`）  
- 支援離散式與 Gaussian 權重策略  

### 預期加入

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

## 函數與參數說明

### smooth_method()

#### main Parameters
| Name | Type | Description |
|:----:|:----:|-------------|
| x | list | X-axis data (e.g., temperature) |
| y | list | Y-axis data (e.g., resistivity) |
| bins | list | X positions defining local data segments used as input to `smooth_method` for smooth filtering |

#### Advanced Parameters
| Name | Type | Default | Description |
|:----:|:----:|:----:|-------------|
| std_num | int | 8 | Threshold on the number of data points in a local segment to enable smooth filtering |
| std_const | float | 1.6 | Standard deviation threshold for retaining data points during the smooth filtering |
| smooth_level | int | 2 | Number of neighboring segments used as a reference to enhance local segment selection robustness |
| min_num | int | 3 | Threshold for discarding local segments with insufficient data points |
| kernel | str | discrete | Defining weights across smooth levels: discrete or gaussian |
| iterate | int | 5 | Upper limit on the number of iterations |
| sample_max | int | 1000 | Upper limit on the number of data points used for regression |

#### Returns ( *pandas.DataFrame* )

| Column | Type | Description |
|:------:|:----:|-------------|
| x | float | X-axis data |
| y | float | Y-axis data |
| tag | int | classification of local segmaent |
| keep | bool | Keep data points that pass smooth-based data cleaning |

---
<br>

### slope_method()

#### main Parameters
| Name | Type | Description |
|:----:|:----:|-------------|
| x | list | X-axis data (e.g., temperature) |
| y | list | Y-axis data (e.g., resistivity) |
| bins | list | X positions defining local data segments used for slope calculation |


#### Advanced Parameters
| Name | Type | Default | Description |
|:----:|:----:|:----:|-------------|
| interval | float | 0.75 | Local x-interval size centered at each bin for data selection |
| min_num | int | 10 | Threshold on the number of data points required for local slope estimation |
| interval_min_rate | float | 0.8 | Skip slope calculation if the x-range of a local segment is smaller than interval_min_rate × (2 × interval) |
| strategy | str | bin | Strategy for selecting the x-value of each local slope (bin or mean) |
| bin_shift_rate | float | 0.3 | Maximum allowed deviation between bin and mean (relative to interval) for slope calculation |

#### Returns ( *pandas.DataFrame* )

| Column | Type | Description |
|:------:|:----:|-------------|
| x | float | Representative x-value of the local segment |
| slope | float | Estimated local slope |
| slope_std | float | Uncertainty of the slope estimat |
| intercept | float | Estimated intercept |
| intercept_std | float | Uncertainty of the intercept estimate |
| r | float | Pearson correlation coefficient of the local segment |

```
def smooth_method(
    x:list, 
    y:list, 
    bins:list, 
    std_num:int=8, 
    std_const:float=1.6, 
    smooth_level:int=2, 
    iterate:int=5, 
    sample_max:int=1000, 
    min_num:int=3, 
    kernel="discrete"
    ) -> pd.DataFrame:
```