
from flask import Flask, request,jsonify, render_template, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
import pandas as pd
from scipy.io.arff import loadarff
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import io
import base64
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.io import arff
from sklearn.metrics import f1_score  # Importer f1_score
from io import BytesIO


# Créer une instance de Flask
app = Flask(__name__)

# Configurer les dossiers de téléchargement et d'images
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['IMAGE_FOLDER'] = 'static/images'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMAGE_FOLDER'], exist_ok=True)

# Configurer la clé secrète pour les messages flash
app.config['SECRET_KEY'] = 'votre_cle_secrete'

# Variables globales pour stocker les parties training et testing
training_part = None
testing_part = None
highest_accuracy_model = None

# Route pour la page d'accueil
@app.route('/', methods=['GET', 'POST'])
def index():
    global training_part, testing_part  
    train_html = ''
    test_html = ''
    train_rows = 0
    test_rows = 0
    # Permet d'utiliser les variables globales
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.arff'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Charger le fichier .arff avec loadarff de scipy
            data, meta = loadarff(filepath)
            df = pd.DataFrame(data)
              # Vérifiez si le dataset est vide
            if df.empty:
                flash('The dataset is empty , Please upload another Dataset')
                return redirect(url_for('index'))
            file_name = file.filename
            # Obtenir les informations sur le dataset
            summary = {
                'file_name': file_name, 
                'rows': df.shape[0],  # Nombre de lignes
                'columns': df.shape[1],  # Nombre de colonnes
                'attributes': df.columns.tolist()  # Liste des attributs
            }
            
            # Convertir toutes les lignes du dataset en HTML
            all_rows_html = df.head(100000).to_html(classes='table')
             # Calculer les distributions de fréquence des colonnes catégoriques
            frequency_distributions = []
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns
            
            for col in categorical_columns:
                freq_dist = df[col].value_counts().to_frame().reset_index()
                freq_dist.columns = [col, 'Frequency']
                frequency_distributions.append(freq_dist.to_html(classes='table'))
            
            
             
            if 'split' in request.form:
     
                training_size = int(request.form.get('training_size', 80))
                testing_size = int(request.form.get('testing_size', 20))                
                if training_size + testing_size != 100:
                  flash('The sum of training and testing percentages must be equal to 100.')
                  return redirect(url_for('index'))
                # Diviser les données en ensembles d'entraînement et de test selon les pourcentages spécifiés
                training_part, testing_part = train_test_split(df, test_size=testing_size / 100, train_size=training_size / 100, random_state=42)
    
                # Obtenir les nombres de lignes des deux parties
                train_rows = training_part.shape[0]
                test_rows = testing_part.shape[0]
                # Inverser les lignes et les colonnes des DataFrames (transposer)
                train_df_transposed = training_part.transpose()
                test_df_transposed = testing_part.transpose()
                
                train_columns_transposed = train_df_transposed.iloc[:, :50000]
                test_columns_transposed = test_df_transposed.iloc[:, :50000]
                # Convertir les DataFrames en HTML
                train_html = train_columns_transposed.head(50).to_html(classes='table')
                test_html = test_columns_transposed.head(50).to_html(classes='table')
                
                # Rendre le template `split.html` avec les données
                return render_template('index.html', train_rows=train_rows, test_rows=test_rows, train_html=train_html, test_html=test_html)
          
             # Appelez la fonction pour créer les diagrammes circulaires
            pie_charts = plot_pie_charts(df)
            # Visualiser la distribution des colonnes numériques
            dist_plot = plot_distributions(df)
            return render_template('summary.html', summary=summary, instances=all_rows_html, dist_plot=dist_plot, pie_charts=pie_charts)
        else:
            flash('Please upload a file Or check file type ')
            return redirect(url_for('index'))
    
    return render_template('index.html')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64

def plot_distributions(df):
    # Filtrer les colonnes numériques
    numeric_columns = df.select_dtypes(include=['number']).columns
    
    # Définir le nombre de colonnes par ligne à 3 (vous pouvez ajuster selon vos besoins)
    ncols = len(numeric_columns)
    
    # Créer une figure avec une seule ligne et `ncols` colonnes
    fig, axes = plt.subplots(ncols=ncols, figsize=(ncols * 6, 4))
    
    # Tracer les histogrammes pour chaque colonne numérique
    for i, col in enumerate(numeric_columns):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
    
    # Ajuster la disposition et l'espacement
    plt.tight_layout()
    
    # Enregistrer la figure dans un objet BytesIO
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    
    # Convertir l'image en base64
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    
    # Fermer la figure
    plt.close(fig)
    
    # Retourner l'image encodée en base64
    return f"data:image/png;base64,{img_base64}"

import matplotlib.pyplot as plt
import pandas as pd
import io
import base64

def plot_pie_charts(df):
    # Filtrer les colonnes catégoriques
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    # Créer une liste pour stocker les images base64 des diagrammes circulaires
    pie_charts = []
    
    # Couleurs pour les sections du diagramme circulaire (bleu et vert)
    colors = ['#ca79c2e7', '#86c7e1']
    
    # Parcourir chaque colonne catégorique
    for col in categorical_columns:
        # Calculer les fréquences de la colonne
        frequencies = df[col].value_counts()
        
        # Créer un diagramme circulaire pour la colonne
        fig, ax = plt.subplots()
        ax.pie(frequencies, labels=frequencies.index, autopct='%1.1f%%', colors=colors)
        ax.set_title(f'Pie Chart of {col}')
        
        # Enregistrer l'image du diagramme circulaire en base64
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png')
        img_bytes.seek(0)
        img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
        plt.close(fig)
        
        # Ajouter l'image base64 à la liste
        pie_charts.append(f"data:image/png;base64,{img_base64}")
    
    return pie_charts





@app.route('/train', methods=['POST'])
def train():
    global training_part, testing_part  # Utilisez les variables globales pour accéder aux données d'entraînement et de test
    global highest_accuracy_model 
    if training_part is None:
     flash('No training data found. Please split the data first.')
     return redirect(url_for('index'))
    
    # Obtenir la liste des classificateurs à entraîner
    selected_classifiers = request.form.getlist('classifier')
    select_all = request.form.get('select_all')
    # Liste des classificateurs disponibles
    classifiers = {
        'decision_tree': DecisionTreeClassifier(),
        'naive_bayes': GaussianNB(),
        'random_forest': RandomForestClassifier(),
        'svm': SVC()
    }
     # If the user selected "select all" option, choose all available classifiers
    if select_all:
        selected_classifiers = list(classifiers.keys())
    
    results = []
    total_accuracy = 0
   
    highest_accuracy = 0
    
    # Entraîner chaque classificateur sélectionné et calculer accuracy et f1-score
    for clf_name in selected_classifiers:
        if clf_name in classifiers:
            clf = classifiers[clf_name]
            
            # Diviser les données d'entraînement en fonctionnalités (X) et étiquettes (y)
            X_train = training_part.iloc[:, :-1]
            y_train = training_part.iloc[:, -1]
            
            # Si les étiquettes sont de type chaîne, les convertir en valeurs discrètes
            if y_train.dtype == 'object':
                y_train = pd.factorize(y_train)[0]
            
            # Entraîner le modèle
            clf.fit(X_train, y_train)
            
            # Prédire sur les données de test
            y_test = testing_part.iloc[:, :-1]
            y_test_labels = testing_part.iloc[:, -1]
            
            if y_test_labels.dtype == 'object':
                y_test_labels = pd.factorize(y_test_labels)[0]
            
            y_pred = clf.predict(y_test)
            
            # Calculer l'accuracy et le f1-score
            accuracy = accuracy_score(y_test_labels, y_pred)
            f1 = f1_score(y_test_labels, y_pred, average='weighted')
            
            # Ajouter les résultats au tableau
            results.append({
                'classifier': clf_name,
                'accuracy': accuracy,
                'f1_score': f1
            })
             # Update the highest accuracy model
            if accuracy > highest_accuracy:
             highest_accuracy = accuracy
             highest_accuracy_model = {
                  'classifier': clf_name,
                   'classifier_instance': clf,
                   'accuracy': accuracy,
                        'f1_score': f1
                    }
            # Accumuler l'accuracy pour calculer les poids
            total_accuracy += accuracy
    
    # Calculer les poids de chaque classificateur si l'utilisateur choisit "calculer automatiquement les poids"
    weight_option = request.form.get('weight_option')
    
    if weight_option == 'automatic':
        for result in results:
            result['weight'] = result['accuracy'] / total_accuracy
    elif weight_option == 'manual':
        # Si l'option manuelle est sélectionnée, utilisez les poids fournis par l'utilisateur
        for result in results:
            weight_key = f'weight_{result["classifier"]}'
            weight_value = (request.form.get(weight_key, 0.0))
            result['weight'] = weight_value
    
    # Rendre les résultats en tant que JSON pour l'affichage dans le frontend
    return jsonify(results)

@app.route('/get_highest_accuracy_model', methods=['POST'])
def get_highest_accuracy_model():
    # Vérifiez si le modèle ayant la meilleure précision existe
    if not highest_accuracy_model:
        # Si aucun modèle n'est disponible, renvoyer un message d'erreur
        response = {
            'status': 'error',
            'message': 'No trained models found. Please train the models first.'
        }
        return jsonify(response), 404

    # Structurez les détails du modèle dans un dictionnaire
    model_details = {
        'classifier': highest_accuracy_model['classifier'],
        'accuracy': highest_accuracy_model['accuracy'],
        'f1_score': highest_accuracy_model['f1_score']
    }

    # Renvoyez les détails du modèle sous forme de JSON
    response = {
        'status': 'success',
        'model': model_details
    }
    return jsonify(response), 200




@app.route('/test_model', methods=['POST'])
def test_model():
    global testing_part  # Utilisez la partie de test globale
    global highest_accuracy_model  # Utilisez le classificateur ayant la plus grande précision
    
    # Vérifiez si le modèle ayant la plus grande précision existe
    if not highest_accuracy_model:
        response = {
            'status': 'error',
            'message': 'No highest accuracy model found. Please train the models first.'
        }
        return jsonify(response), 404
    
    # Vérifiez si la partie de test existe
    elif testing_part is None:
        response = {
            'status': 'error',
            'message': 'No testing data found. Please split the data first.'
        }
        return jsonify(response), 404
    
    # Initialiser une liste pour stocker les résultats
    results = []
    
    # Utilisez le classificateur avec la plus grande précision
    classifier_instance = highest_accuracy_model['classifier_instance']
    
    # Divisez les données de test en fonctionnalités (X) et étiquettes (y)
    X_test = testing_part.iloc[:, :-1]
    y_test = testing_part.iloc[:, -1]
    
    # Si les étiquettes sont de type chaîne, les convertir en valeurs discrètes
    if y_test.dtype == 'object':
        y_test, _ = pd.factorize(y_test)
    
    # Prédire les étiquettes pour chaque instance dans les données de test
    y_pred = classifier_instance.predict(X_test)
    
    # Parcourez les instances de test en utilisant leurs indices
    for idx in range(len(X_test)):
        # Obtenez l'index de l'instance actuelle
        index = X_test.index[idx]
        
        # Obtenez les valeurs des étiquettes réelle et prédite
        real_label = int(y_test[idx])  # Convertir en type natif
        predicted_label = int(y_pred[idx])  # Convertir en type natif
        
        # Ajoutez les résultats à la liste
        results.append({
            'index': int(index),  # Convertir l'index en type natif
            'real_label': real_label,
            'predicted_label': predicted_label
        })
    
    # Retournez les résultats sous forme de JSON pour l'affichage dans le frontend
    return jsonify(results), 200





import base64

# Convertir un objet bytes en une chaîne base64
def bytes_to_base64_str(data):
    return base64.b64encode(data).decode('utf-8')


@app.route('/get_instance_details', methods=['POST'])
def get_instance_details():
    data = request.get_json()
    index = data.get('index')
    
    if testing_part is not None and 0 <= index < len(testing_part):
        # Obtenir l'instance spécifiée
        instance = testing_part.iloc[index]

        # Si instance contient des données de type bytes, les convertir
        instance_details = instance.to_dict()
        for key, value in instance_details.items():
            if isinstance(value, bytes):
                # Convertir les bytes en chaîne base64
                instance_details[key] = base64.b64encode(value).decode('utf-8')

        return jsonify(instance_details), 200
    else:
        # Index invalide ou données de test non définies
        return jsonify({'error': 'Invalid index or testing data not available'}), 400



@app.route('/reset', methods=['POST'])
def reset():
    global training_part, testing_part, highest_accuracy_model

    # Reset global variables or any other application state
    training_part = None
    testing_part = None
    highest_accuracy_model = None

    # Optionally, clear the Flask session
    session.clear()

    # Redirect the user back to the main page
    return redirect(url_for('index')+'#data-upload-and-split')



import numpy as np

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import precision_score, recall_score, roc_curve, roc_auc_score, mean_squared_error, mean_absolute_error
import seaborn as sns

@app.route('/predict_and_visualize', methods=['POST'])
def predict_and_visualize():
    global testing_part, highest_accuracy_model

    if not highest_accuracy_model:
        response = {
            'status': 'error',
            'message': 'No highest accuracy model found. Please train the models first.'
        }
        return jsonify(response), 404
    
    if testing_part is None:
        response = {
            'status': 'error',
            'message': 'No testing data found. Please split the data first.'
        }
        return jsonify(response), 404
    
    classifier_instance = highest_accuracy_model['classifier_instance']
    X_test = testing_part.iloc[:, :-1]
    y_test = testing_part.iloc[:, -1]
    
    if y_test.dtype == 'object':
        y_test, _ = pd.factorize(y_test)
    
    y_pred = classifier_instance.predict(X_test)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Convert confusion matrix to base64 image
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    plt.close(fig)

    # Calculate classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Calculate additional metrics
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Calculate ROC curve and AUC if applicable (for binary or multiclass classification)
    roc_curve_data = None
    auc_score = None
    if len(np.unique(y_test)) <= 2:  # Check if binary or multiclass
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred)
        
        # Convert ROC curve to base64 image
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        
        roc_img_bytes = io.BytesIO()
        plt.savefig(roc_img_bytes, format='png')
        roc_img_bytes.seek(0)
        roc_img_base64 = base64.b64encode(roc_img_bytes.read()).decode('utf-8')
        plt.close(fig)
        
        roc_curve_data = f"data:image/png;base64,{roc_img_base64}"

    # Prepare results
    results = {
        'confusion_matrix': f"data:image/png;base64,{img_base64}",
        'classification_report': class_report,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }
    
    if roc_curve_data and auc_score:
        results['roc_curve'] = roc_curve_data
        results['auc_score'] = auc_score
    
    # Renvoyer les résultats sous forme de JSON
    return jsonify(results), 200








if __name__ == '__main__':
    app.run(debug=True)