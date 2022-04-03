import pandas
import seaborn
import matplotlib.pyplot as plt

data = pandas.read_csv("data/penguins.csv")
seaborn.displot(data, y='flipper_length_mm', x='species')
plt.show()
