import pandas as pd

import algorithm_apriori


def extract_country_basket(data, country):
    return (data[data['Country'] == country]
            .groupby(['InvoiceNo', 'Description'])['Quantity']
            .sum().unstack().reset_index().fillna(0)
            .set_index('InvoiceNo'))


def unit_parser(value):
    if value <= 0:
        return 0
    if value >= 1:
        return


country_to_experiment = 'United Kingdom'
dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]
frequent_itemset = {}
min_support = 10

# funkcja pobiera dane z serwisu
csv_data = pd.read_excel('http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx')

# formatowanie danych
csv_data['Description'] = csv_data['Description'].str.strip()  # usuniecie niepotrzebnych spacji
csv_data.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
csv_data['InvoiceNo'] = csv_data['InvoiceNo'].astype('str')  # usuniecie transakcji bez numeru faktury
csv_data = csv_data[~csv_data['InvoiceNo'].str.contains('C')]  # usniecie transakacji karta kredytowa

# wyodrebnienie transakcji dokonanych w Wielkiej Brytanii
country_basket = extract_country_basket(csv_data, country_to_experiment)
# stworzenie tabelii obserwacji
observation_array = country_basket.applymap(unit_parser)

print(algorithm_apriori.apriori(min_support))
