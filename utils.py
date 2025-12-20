import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import netCDF4 as nc


def read_nc(file_path:str, group_name, band_names = ['L_TOA_443', 'L_TOA_490', 'L_TOA_555', 'L_TOA_660', 'L_TOA_865']):
    data = []
    for band_name in band_names:
        band_data = xr.open_dataset(file_path, group=group_name)[band_name]
        data.append(band_data)
        data_con=xr.concat(data, dim='band')
        data_con = data_con.where(data_con != 0, np.nan)
    return data_con 

if __name__ == "__main__":
    bands = [443, 490, 555, 660, 865]
    band_names = [f"L_TOA_{band}" for band in bands]
    print(band_names)

    file_path = '/Users/zy/Python_code/My_Git/img_match/SR_Imagery/GK2_GOCI2_L1B_20250504_021530_LA_S007.nc'
    geo_group = 'geophysical_data'
    nav_group = 'navigation_data'

    data = read_nc(file_path, geo_group, band_names)

    print(f"data: {data}")                # 打印基本结构和维度
    ''' 
    print(f"data.shape: {data.shape}")          # 打印 shape
    print(f"data.dims: {data.dims}")           # 打印维度名称
    print(f"data.coords: {data.coords}")         # 打印坐标信息
    print(f"data.attrs: {data.attrs}")          # 打印属性信息
    '''

    plt.figure()
    plt.imshow(data[0, :, :],vmin=60,vmax=100)  # 显示
    plt.colorbar()
    plt.show()