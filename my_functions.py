import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import math
import random

def download_file(url):
  os.system('mkdir datasets')
  os.system('cd datasets')
  os.system('wget ' + url)

def euclidean_distance(a:list, b:list) -> float:
  if len(a) != len(b):
    raise ValueError('{} len is diferent from {} len'.format(str(a), str(b)))
  s = 0
  for i in range(len(a)):
    s += (a[i] - b[i])**2
  return s**0.5

def split_train_test(dataset=pd.DataFrame, train_size=0.8):
  # set train size
  dataset_len = len(dataset)
  train_len = int(dataset_len*train_size)
  # split train and test
  train, test = dataset[:train_len], dataset[train_len:]
  return train, test

def interaction(dataset:pd.DataFrame, model, train_size=0.8, label='label') -> tuple((list, list)):
  # shuffle dataset
  dataset = dataset.sample(frac=1)
  # split train and test
  train, test = split_train_test(dataset, train_size)
  # Train:
  model.fit(train.drop(columns=[label]),list(train[label]))
  # Test:
  predictions = model.predict(test.drop(columns=[label]))
  actual_values = list(test[label])
  return predictions, actual_values

def compute_hit_rates(confusion_matrixs=[]):
  hit_rates = []
  total = sum(sum(confusion_matrixs[0]))
  for cm in confusion_matrixs:
    hits = 0
    for i in range(len(confusion_matrixs[0])):
        hits += cm[i][i]
    hit_rates.append((hits/total))
  return hit_rates

def compute_mean(number_list:list):
  try:
    if len(number_list) == 0:
      return 0
    return sum(number_list) / len(number_list)
  except:
    print(number_list)

# Standard Deviation or Desvio pradrão
def compute_standard_deviation(hit_rates=[]):
  hit_rates_mean = compute_mean(hit_rates)
  s = 0
  for hit_rate in hit_rates:
    s += (hit_rate - hit_rates_mean)**2
  return (s/(len(hit_rates)-1))**0.5

def plot_bar_graficy(dict_2d={'K':[], 'accuracy':[]}):
  df = pd.DataFrame(dict_2d)
  # set rotation e fontsize for xticks
  plt.xticks(rotation=75,fontsize=7)
  # set data end axis
  [x, y] = dict_2d.keys()
  ax = sns.barplot(data=df, y=y, x=x)
  # plot
  plt.show()

def confusion_matrix(labels:list, test:list, pred:list):
  labels_len = len(labels)
  # create a matrix labels_len x labels_len
  confusion_matrix = [
      [0 for __ in range(labels_len)] for _ in range(labels_len)
    ]
  # compute confusion matrix
  for i in range(len(test)):
    te_label_i = labels.index(test[i])
    pr_label_i = labels.index(pred[i])
    confusion_matrix[te_label_i][pr_label_i] += 1
  # return confusion matrix 
  return confusion_matrix 

def plot_confusion_matrix(dataset_labels, confusion_matrix, title='Confusion Matrix\n\n'):
  heatmap = sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
  # set map title and Labels
  heatmap.set_title(title)
  heatmap.set_xlabel('\nPredicted Values')
  heatmap.set_ylabel('Actual Values')
  # set dataset labels
  heatmap.xaxis.set_ticklabels(dataset_labels)
  heatmap.yaxis.set_ticklabels(dataset_labels)
  # plot
  plt.show()

def normalize_data(dataset:pd.DataFrame, features:list):
    # for each feature
    for feature in features:
        serie = dataset[feature]
        # find max and min value
        feat_max = max(serie)
        feat_min = min(serie)
        # rescale each valeu from the dataset, scale(0 - 1)
        new_values = []
        for i in range(len(serie)):
            value = serie[i]
            new_values.append((value - feat_min) / (feat_max - feat_min))
        dataset[feature] = new_values
    return dataset


# Decision boundary or Decision Surface or "Superfície de decisão"
def decision_surface(model, dataset:pd.DataFrame, features:list, label:str):
    # get selected features
    dataset_sample = dataset[[*features, label]]
    # shuffle, split train and test
    train, test = split_train_test(dataset_sample.sample(frac=1))
    # train model
    X = train.drop(columns=[label])
    y = list(train[label])
    model.fit(X, y)
    # set step for the 
    column1 = dataset_sample[features[0]]
    column2 = dataset_sample[features[1]]
    column1_max = max(column1)
    column2_max = max(column2)
    column1_steps = np.arange(0, column1_max, column1_max/100)
    column2_steps = np.arange(0, column2_max, column2_max/50)
    # create surface
    decision_surface = pd.DataFrame(
        columns=[
            features[0],
            features[1]
        ])
    i = len(dataset)
    for c1 in column1_steps:
        for c2 in column2_steps:
            decision_surface = pd.concat([
            decision_surface,
            pd.DataFrame({
                features[0]: [c1],
                features[1]: [c2]
            })
            ], ignore_index=True)
    # predict surface
    pred = model.predict(decision_surface)
    # add a new column 'Class' with the prediction
    decision_surface[label] = pred
    # Modificy train and test labels, ex. label -> label_test and label_train
    test[label] = test[label].apply(lambda label: str(label) + '_test')
    train[label] = train[label].apply(lambda label: str(label) + '_train')
    # Merge decision_surface, test and train
    decision_surface = pd.concat([decision_surface, test, train], ignore_index=True)
    return decision_surface

def plot_scatter(df:pd.DataFrame, columns:list, label='label', title='Scatter'):
    # plot
    scatter = sns.scatterplot(x=columns[0], y=columns[1], hue=label,
                                data=df)
    sns.move_legend(scatter, "upper left", bbox_to_anchor=(1, 1))
    # set map title and Labels
    scatter.set_title(title + '\n\n')
    scatter.set_xlabel('\n' + columns[0])
    scatter.set_ylabel(columns[1])
    plt.show()

def matrix_distance(a:np.atleast_2d, b:np.atleast_2d):
    dim = a.shape[0]
    return euclidean_distance(a.reshape(dim**2,1), b.reshape(dim**2,1))

# Compute closest confusion matrix from the mean
def mean_closest_quadratic_matrix(matrix_list:list):
    dim = np.array(matrix_list[0]).shape[0]
    # compute quadratic matrix mean
    m_mean = sum(matrix_list) / len(matrix_list)
    # find the closest mean to the mean
    closest_m_i = 0
    closest_m_distance = matrix_distance(m_mean, matrix_list[0])
    for i in range(1, len(matrix_list)):
        crr_distance = matrix_distance(m_mean, matrix_list[i])
        if crr_distance < closest_m_distance:
          closest_m_i = i
          closest_m_distance = crr_distance
    return matrix_list[closest_m_i]

# Compute closest confusion matrix from the mean
def mean_closest_confusion_matrix(confusion_matrixs:list, dataset_labels:list):
    # compute the interactions mean
    cm_mean = sum(confusion_matrixs) / len(confusion_matrixs)
    # find the closest cm to the mean
    def cm_distance(a, b, dataset_labels):
        labels_len = len(dataset_labels)
        return euclidean_distance(a.reshape((labels_len**2,1)),
            b.reshape((labels_len**2,1)))
    closest_cm = {
        'i': 0,
        'distance': cm_distance(cm_mean, confusion_matrixs[0], dataset_labels) 
    }
    for i in range(1, len(confusion_matrixs)):
        distance = cm_distance(cm_mean, confusion_matrixs[i], dataset_labels)
        if distance < closest_cm['distance']:
            closest_cm['distance'] = distance
            closest_cm['i'] = i
    return confusion_matrixs[closest_cm['i']]


def find_closest_neighbor(point:list, neighbors:list):
    closest_index = 0
    closest_distance = euclidean_distance(point, neighbors[0])
    for i, neighbor in enumerate(neighbors[1:]):
        crr_distance = euclidean_distance(point, neighbor)
        if crr_distance < closest_distance:
            closest_distance = crr_distance
            closest_index = i + 1
    return closest_index

def compute_moda(label_list):
    if len(label_list) < 1:
        return None
    # count label frequency
    labels = {}
    for i in label_list:
        if labels.__contains__(i):
            labels[i] += 1
        else:
            labels[i] = 1
    # find the most frequent label
    labels_keys = list(labels.keys())
    label = labels_keys[0]
    n = labels[label]
    for key in labels_keys:
        crr_n = labels[key]
        if crr_n > n:
            n = crr_n
            label = key
    return label

def plot_acc_std_closest_cm_to_the_mean(df:pd.DataFrame, label:str, Model, params=[], model_name=''):
    cms = []
    dataset_unique_labels = list(df[label].unique())
    for _ in range(20):
        model = Model(*params)
        pred, actu = interaction(df, model, label=label)
        cm = confusion_matrix(dataset_unique_labels, actu, pred)
        cms.append(np.array(cm))
    # compute hit rates
    hit_rates = compute_hit_rates(cms)
    # compute accuracy
    acc = compute_mean(hit_rates)
    # compute standard deviation
    std = compute_standard_deviation(hit_rates)
    acc_std_str = 'Accuracy: {:.2};  Standard Deviation: {:.3}\n\n'.format(acc, std)
    # Using the cloasest case, to the mean
    closest_cm = mean_closest_confusion_matrix(cms, dataset_unique_labels)
    # plot accuracy and standard deviation and the selected interaction
    plot_confusion_matrix(
      dataset_unique_labels,
      closest_cm, 
      title=model_name + '\n' + acc_std_str + 'Confusion Matrix\n'
    )

def plot_surface_3D(df:pd.DataFrame, x:str, y:str, z:str):
    sns.set(style = "darkgrid")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    x = df[x]
    y = df[y]
    z = df[z]

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)

    ax.plot_surface(x, y, z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

    sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1))

    plt.show()
  
def plot_scatter_3D(df:pd.DataFrame, x:str, y:str, z:str):
    sns.set(style = "darkgrid")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    x = df[x]
    y = df[y]
    z = df[z]

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)

    ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5)

    # sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1))

    plt.show()
  
def only_diagonal_matrix(matrix:np.atleast_2d):
    new_matrix = np.zeros((matrix.shape))
    for i in range(matrix.shape[0]):
        new_matrix[i][i] = matrix[i][i]
    return new_matrix
 
def compute_covariance_matrix(values:np.array):
    n_rows, n_cols =values.shape
    cov = np.zeros((n_cols, n_cols))
    for x in range(n_cols):
        mean_x = compute_mean(values[:, x])
        for y in range(n_cols): 
            mean_y = compute_mean(values[:, y])
            cov[x, y] = np.sum((values[:, x] - mean_x) * (values[:, y] - mean_y)) / (n_rows - 1)
    return np.array(cov)

def compute_selected_feats(dataset:pd.DataFrame, column_label:str):
  dataset_values = dataset.drop(columns=[column_label]).values
  eig_values,_ = np.linalg.eig(compute_covariance_matrix(dataset_values))
  most_high_v = [-math.inf, -math.inf]
  most_high_i = [-1, -1]
  for i, v in enumerate(eig_values):
      if v > most_high_v[0]:
          most_high_v[0] = v
          most_high_i[0] = i
      elif v > most_high_v[1]:
          most_high_v[1] = v
          most_high_i[1] = i

  feats = dataset.drop(columns=[column_label]).columns
  selected_feats = [feats[i] for i in most_high_i]
  return selected_feats