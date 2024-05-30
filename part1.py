import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Citirea datelor
df = pd.read_csv('train.csv')

#1
# Numarul de coloane
num_cols = len(df.columns)
print("Numarul de coloane:", num_cols)

# Numarul de linii
num_rows = len(df)
print("Numarul de linii:", num_rows)

# Tipurile datelor din fiecare coloana
data_types = df.dtypes
print("Tipurile datelor din fiecare coloana:")
print(data_types)

# Numarul de valori lipsa pentru fiecare coloana
missing_values = df.isnull().sum()
print("Numarul de valori lipsa pentru fiecare coloana:")
print(missing_values)

# Verificarea existenței de linii duplicate
num_duplicates = df.duplicated().sum()
print("Numarul de linii duplicate:", num_duplicates)

#2
# Procentul persoanelor care au supravietuit si care nu au supravietuit
survived_percent = df['Survived'].value_counts(normalize = True) * 100

# Procentul pasagerilor pentru fiecare tip de Clasa (Pclass)
class_percent = df['Pclass'].value_counts(normalize = True) * 100

# Procentul bărbatilor si femeilor
gender_percent = df['Sex'].value_counts(normalize = True) * 100

# Crearea unui grafic
fig, axes = plt.subplots(1, 3, figsize = (15, 5))
survived_percent.plot(kind = 'bar', ax = axes[0], title = 'Supravietuire')
class_percent.plot(kind = 'bar', ax = axes[1], title = 'Clasa')
gender_percent.plot(kind = 'bar', ax = axes[2], title = 'Sex')
plt.tight_layout()
plt.savefig('grafic(task2).png')
plt.close()

#3
# Generarea histogramelor
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
num_plots = len(numeric_cols)
fig, axes = plt.subplots(4, ncols = 2, figsize = (15, 5 * 4))
axes = axes.flatten()

for ax, col in zip(axes, numeric_cols):
    sns.histplot(df[col].dropna(), bins = 20, kde = True, ax = ax)
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel('Frecventa')

# Elimin subloturile nefolosite
for i in range(len(numeric_cols), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig('all_histograms.png')
plt.close(fig)

#4
# Identificarea coloanelor cu valori lipsa
missing_cols = df.columns[df.isnull().any()]
print("Coloanele cu valori lipsa:", missing_cols)

# Calculul Numarului si procentului de valori lipsa pentru fiecare coloana si Clasa
missing_data = df[missing_cols].isnull().sum()
missing_percentage = (missing_data / len(df)) * 100
print("Numarul de valori lipsa pentru fiecare coloana si procentul lor:")
print(missing_data)
print("Procentul valorilor lipsa pentru fiecare coloana si Clasa:")
print(missing_percentage)

#5
# Determinarea Numarului de pasageri in fiecare categorie de varsta
df['Age_Category'] = pd.cut(df['Age'], bins = [0, 20, 40, 60, df['Age'].max()], labels = ['0-20', '21-40', '41-60', '61-max'])
age_category_counts = df['Age_Category'].value_counts()

# Crearea unui grafic pentru a evidenția aceste rezultate
plt.figure(figsize = (10, 6))
age_category_counts.plot(kind = 'bar', title = 'Numarul de pasageri in fiecare categorie de varsta')
plt.xlabel('Categorie de varsta')
plt.ylabel('Numarul de pasageri')
plt.tight_layout()
plt.savefig('grafic(task5).png')
plt.close()

#6
# Determinarea procentului de Supravietuire pentru fiecare categorie de varsta
survival_percentage_age = df.groupby('Age_Category')['Survived'].mean() * 100

# Crearea unui grafic pentru a evidenția aceste rezultate
plt.figure(figsize = (10, 6))
survival_percentage_age.plot(kind = 'bar', title = 'Procentul de Supravietuire pentru fiecare categorie de varsta')
plt.xlabel('Categorie de varsta')
plt.ylabel('Procentul de Supravietuire')
plt.tight_layout()
plt.savefig('grafic(task6).png')
plt.close()

#7
# Determinarea procentului de copii la bord
children_percent = (df['Age'] < 18).mean() * 100

# Calcularea ratei de Supravietuire pentru copii si adulți
survival_children_adults = df.groupby(df['Age'] < 18)['Survived'].mean() * 100

# Crearea unui grafic pentru a evidenția aceste rezultate
plt.figure(figsize = (10, 6))
survival_children_adults.plot(kind = 'bar', title = 'Rata de Supravietuire pentru copii si adulți')
plt.xlabel('Categorie de varsta')
plt.ylabel('Rata de Supravietuire')
plt.xticks([False, True], ['Adulți', 'Copii'])
plt.tight_layout()
plt.savefig('grafic(task7).png')
plt.close()

#8
# Completarea valorilor lipsa pentru varsta cu media varstei pasagerilor din aceeasi clasa
df['Age'] = df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.mean()))

# Completarea celor mai frecvente valori pentru coloanele categoriale
for col in df.select_dtypes(include = ['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

df.to_csv('titanic_modificat(task8).csv', index = False)

#9
# Extrage titlurile din coloana "Name"
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.')

# Verificarea dacă titlurile corespund sexului persoanelor
title_gender_check = df.groupby(['Title', 'Sex']).size().unstack()

# Realizarea graficului
plt.figure(figsize = (10, 6))
sns.countplot(data = df, x = 'Title', hue = 'Sex')
plt.title('Distributia titlurilor in functie de sex')
plt.xlabel('Titlu')
plt.ylabel('Numar de persoane')
plt.legend(title = 'Sex')
plt.tight_layout()
plt.savefig('grafic(task9).png')
plt.close()

#10
# Histograma pentru investigarea relației dintre tarif, Clasa si Supravietuire
plt.figure(figsize = (10, 6))
sns.catplot(data = df.head(100), x = 'Pclass', y = 'Fare', hue = 'Survived', kind = 'swarm')
plt.title('Relatia intre tarif, Clasa si Supravietuire')
plt.xlabel('Clasa')
plt.ylabel('Tarif')
plt.tight_layout()
plt.savefig('grafic(task10).png')
plt.close()