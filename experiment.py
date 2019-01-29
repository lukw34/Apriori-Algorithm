import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import sys


def extract_country_basket(data, country):
    return (data[data['Country'] == country]
            .groupby(['InvoiceNo', 'Description'])['Quantity']
            .sum().unstack().reset_index().fillna(0)
            .set_index('InvoiceNo'))


def unit_parser(value):
    if value <= 0:
        return 0
    if value >= 1:
        return 1


pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)
country_to_experiment = sys.argv[2]
min_support = float(sys.argv[1])

# funkcja pobiera dane z serwisu
csv_data = pd.read_excel('http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx')

# formatowanie danych
csv_data['Description'] = csv_data['Description'].str.strip()  # usuniecie niepotrzebnych spacji
csv_data.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
csv_data['InvoiceNo'] = csv_data['InvoiceNo'].astype('str')  # usuniecie transakcji bez numeru faktury
csv_data = csv_data[~csv_data['InvoiceNo'].str.contains('C')]  # usniecie transakacji karta kredytowa

# wyodrebnienie transakcji dokonanych we Francji
country_basket = extract_country_basket(csv_data, country_to_experiment)
# stworzenie tabelii obserwacji
observation_array = country_basket.applymap(unit_parser)
observation_array.drop('POSTAGE', inplace=True, axis=1)
frequent_itemset = apriori(observation_array, min_support=min_support, use_colnames=True)
rules = association_rules(frequent_itemset, metric="lift", min_threshold=2).sort_values(by='lift')
for index, row in rules.iterrows():
    antecetends = row['antecedents']
    consequents = row['consequents']
    print('if ', antecetends, ' ---> ', consequents, ' lift: ', row['lift'], 'support: ' + str(row['support']))
