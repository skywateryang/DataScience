import urllib.request, json
import pandas as pd
import colour
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rc('legend', fontsize=12 )

# get json data
with urllib.request.urlopen("https://raw.githubusercontent.com/Ovilia/lipstick/gh-pages/dist/lipstick.json") as url:
    data = json.loads(url.read().decode())
dataframe = pd.json_normalize(data['brands'], ['series', 'lipsticks'], ['name', ['series', 'name']],
                              meta_prefix='brand_')


def HEX_2_XYY(hex):
    return colour.XYZ_to_xyY(colour.sRGB_to_XYZ(colour.notation.HEX_to_RGB(hex)))


def HEX_2_LAB(hex):
    return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(colour.notation.HEX_to_RGB(hex)))

# image1
dataframe.rename(columns={'brand_name': '品牌名'},inplace=True)
dataframe[['x', 'y', 'Y']] = dataframe.apply(lambda x: HEX_2_XYY(x['color']), axis=1, result_type='expand')
dataframe[['lab_L', 'lab_X', 'lab_Y']] = dataframe.apply(lambda x: HEX_2_LAB(x['color']), axis=1, result_type='expand')
dataframe = dataframe[dataframe['x'] > 0.37].reset_index()
colour.plotting.plot_chromaticity_diagram_CIE1931()
# color_x=dataframe['x'].to_list()
# color_y=dataframe['y'].to_list()
# plt.scatter(x=color_x,y=color_y)
sns.scatterplot(x='x', y='y', data=dataframe, hue='品牌名')
plt.xlabel("CIE X", fontsize=20)
plt.ylabel("CIE Y",fontsize=20)
plt.title("色彩分布图",fontsize=30)

# image2
dataframe['品牌-系列名'] = dataframe['品牌名'] + '-' + dataframe['brand_series.name']
meancolor = pd.DataFrame(dataframe.groupby('品牌-系列名')['x', 'y'].mean()).reset_index()
colour.plotting.plot_chromaticity_diagram_CIE1931()
seq = pd.DataFrame(dataframe.groupby('品牌-系列名')['x', 'y'].mean()).reset_index().sort_values(by='x')['品牌-系列名']
sns.scatterplot(x='x', y='y', data=meancolor, hue='品牌-系列名', palette='OrRd', hue_order=seq)
plt.xlabel("CIE X", fontsize=20)
plt.ylabel("CIE Y",fontsize=20)
plt.title("品牌系列-色彩分布",fontsize=30)


# 寻找相似色
most_similar = 99999
most_similar_index = 0
compare_color = HEX_2_LAB('#fd7d36')

for c in range(len(dataframe)):
    color = HEX_2_LAB(dataframe.loc[c, 'color'])
    diff = colour.difference.delta_E_CIE1976(compare_color, color)
    if diff < most_similar:
        most_similar = diff
        most_similar_index = c
        print('diff:{0}, index:{1}'.format(diff, most_similar_index))


# flatui = ["#e27386", "#f55066"]
# sns.palplot(sns.color_palette(flatui))
