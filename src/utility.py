from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def print_accuricy_tree(data_type,depth,x,y,model,size):
    print(f'Model: Decision Tree | Data type: {data_type}')
    print(f'Depth: {depth} | Validation size: {size}%')
    accuracy = accuracy_score(y_true=y, y_pred=model.predict(x))
    print(f'{data_type}, accuracy: {accuracy}')
    nconfusion_matrix = confusion_matrix(y_true=y, y_pred=model.predict(x))
    print(f'confusion matrix:\n{nconfusion_matrix}\n')


def print_accuricy_ANN(data_type,x,y,model,size):
    print(f'Model: Neural Network | Data type: {data_type}')
    print(f'Validation size: {size}%')
    accuracy = accuracy_score(y_true=y, y_pred=model.predict(x))
    print(f'{data_type}, accuracy: {accuracy}')
    nconfusion_matrix = confusion_matrix(y_true=y, y_pred=model.predict(x))
    print(f'confusion matrix:\n{nconfusion_matrix}\n')


def build_graphs_for_KMeans(X_train_DF, Y_train_DF):
    x = X_train_DF.drop(columns=['cp', 'restecg', 'slope', 'ca', 'thal', 'age', 'gender'])
    y = X_train_DF.drop(columns=['trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak'])
   
    x['y']=Y_train_DF
    sns.pairplot(x, hue='y')
    plt.show()
    
    y['y']=Y_train_DF
    sns.pairplot(y, hue='y')
    plt.show()


def neural_network_hidden_layer_graph(Xtrain_ANN, Ytrain_ANN, Xvalid_ANN, Yvalid_ANN):
    train_accs = []
    test_accs = []
    max_range = 100
    
    for hidden_layer_sizes in range(1, max_range):
        print(f'Size of layers: {hidden_layer_sizes}')
        model = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), max_iter=100, activation='relu', learning_rate_init=0.001, random_state=1, verbose=False)
        model.fit(Xtrain_ANN, Ytrain_ANN)
        train_acc = model.score(Xtrain_ANN, Ytrain_ANN)
        train_accs.append(train_acc)
        test_acc = model.score(Xvalid_ANN, Yvalid_ANN)
        test_accs.append(test_acc)
    
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, max_range), train_accs, label='Train')
    plt.plot(range(1, max_range), test_accs, label='Test')
    plt.legend()
    plt.xlabel('Hidden Layer Size')
    plt.ylabel('Accuracy')
    plt.title('Neural Network Hidden Layer Size vs Accuracy')
    plt.show()


def tune_max_depth_and_plot(x_train, y_train, x_valid, y_valid):
    results = pd.DataFrame()
    for max_depth in range(1, 20):
        model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)
        model.fit(x_train, y_train)
        results = results.append({'max_depth': max_depth,
            'train_acc': accuracy_score(y_train, model.predict(x_train)),
            'valid_acc': accuracy_score(y_valid, model.predict(x_valid))}, 
            ignore_index=True)
    
    plt.figure(figsize=(9, 4))
    plt.plot(results['max_depth'], results['train_acc'], marker='o', markersize=4, label='Train accuracy')
    plt.plot(results['max_depth'], results['valid_acc'], marker='o', markersize=4, label='Validation accuracy')
    plt.legend()
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Max Depth Tuning')
    plt.show()
    
    print(results.sort_values('valid_acc', ascending=False))
    return results



def k_fold_cv_with_varying_max_depth(x, y, n_splits=10):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = pd.DataFrame()

    for train_index, val_index in kfold.split(x):
        for max_depth in range(1, 20):
            model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)
            model.fit(x.iloc[train_index], y.iloc[train_index])
            acc = accuracy_score(y.iloc[val_index], model.predict(x.iloc[val_index]))
            results = results.append({'max_depth': max_depth, 'fold': kfold.n_splits, 'acc': acc}, ignore_index=True)
    
    mean_acc = results.groupby('max_depth')['acc'].mean().reset_index().sort_values('acc', ascending=False).head(5)
    std_acc = results.groupby('max_depth')['acc'].std().reset_index().sort_values('acc', ascending=False).head(10)
    
    print(f'Top 5 max_depth by mean accuracy: {mean_acc}\n')
    print(f'Top 10 max_depth by accuracy standard deviation: {std_acc}\n') 
    return results