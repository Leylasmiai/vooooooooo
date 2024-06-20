/*!
* Start Bootstrap - Agency v7.0.12 (https://startbootstrap.com/theme/agency)
* Copyright 2013-2023 Start Bootstrap
* Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-agency/blob/master/LICENSE)
*/
//
// Scripts
// 

document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById('train-models-form');

    form.addEventListener('submit', function (event) {
        event.preventDefault(); // Empêcher la soumission par défaut du formulaire

        // Vérifier la somme des poids si l'option manuelle est sélectionnée
        const weightOption = form.elements['weight_option'].value;
        if (weightOption === 'manual') {
            const weightInputs = [
                parseFloat(form.elements['weight_decision_tree'].value) || 0,
                parseFloat(form.elements['weight_naive_bayes'].value) || 0,
                parseFloat(form.elements['weight_random_forest'].value) || 0,
                parseFloat(form.elements['weight_svm'].value) || 0
            ];

            const totalWeight = weightInputs.reduce((sum, weight) => sum + weight, 0);

            if (Math.abs(totalWeight - 1) > 0.01) {
                alert('La somme des poids doit être égale à 1.');
                return; // Ne pas soumettre le formulaire si la validation échoue
            }
        }

        // Envoyer les données du formulaire à Flask en utilisant fetch
        const formData = new FormData(form);
        fetch(form.action, {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Traitez les données JSON reçues ici
            console.log('Résultats:', data);

            // Vous pouvez afficher les résultats sur le frontend ici
            // Par exemple, vous pouvez mettre à jour une table avec les résultats
            const resultsTable = document.getElementById('results-table').getElementsByTagName('tbody')[0];
            resultsTable.innerHTML = ''; // Vide la table existante

            data.forEach(result => {
                const row = document.createElement('tr');

                const modelCell = document.createElement('td');
                modelCell.textContent = result.classifier;
                row.appendChild(modelCell);

                const accuracyCell = document.createElement('td');
                accuracyCell.textContent = result.accuracy.toFixed(2);
                row.appendChild(accuracyCell);

                const weightCell = document.createElement('td');
                weightCell.textContent = result.weight ? result.weight.toFixed(2) : 'N/A';
                row.appendChild(weightCell);

                const f1Cell = document.createElement('td');
                f1Cell.textContent = result.f1_score.toFixed(2);
                row.appendChild(f1Cell);

                resultsTable.appendChild(row);
            });
        })
        .catch(error => {
            console.error('Erreur lors de la soumission du formulaire:', error);
        });
    });
});




//fin

window.addEventListener('DOMContentLoaded', event => {

    // Navbar shrink function
    var navbarShrink = function () {
        const navbarCollapsible = document.body.querySelector('#mainNav');
        if (!navbarCollapsible) {
            return;
        }
        if (window.scrollY === 0) {
            navbarCollapsible.classList.remove('navbar-shrink')
        } else {
            navbarCollapsible.classList.add('navbar-shrink')
        }

    };

    // Shrink the navbar 
    navbarShrink();

    // Shrink the navbar when page is scrolled
    document.addEventListener('scroll', navbarShrink);

    //  Activate Bootstrap scrollspy on the main nav element
    const mainNav = document.body.querySelector('#mainNav');
    if (mainNav) {
        new bootstrap.ScrollSpy(document.body, {
            target: '#mainNav',
            rootMargin: '0px 0px -40%',
        });
    };

    // Collapse responsive navbar when toggler is visible
    const navbarToggler = document.body.querySelector('.navbar-toggler');
    const responsiveNavItems = [].slice.call(
        document.querySelectorAll('#navbarResponsive .nav-link')
    );
    responsiveNavItems.map(function (responsiveNavItem) {
        responsiveNavItem.addEventListener('click', () => {
            if (window.getComputedStyle(navbarToggler).display !== 'none') {
                navbarToggler.click();
            }
        });
    });

});












// Gestionnaire d'événement pour le formulaire de téléchargement
document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();

    // Créez un objet FormData pour récupérer les données du formulaire
    const formData = new FormData(this);

    // Faites une requête POST vers la route '/upload' de votre serveur Flask
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Si la réponse est correcte, affichez les données dans le tableau
        if (data) {
            const dataTable = document.getElementById('dataTable');

            // Vider le tableau existant
            dataTable.innerHTML = '';

            // Créez les en-têtes du tableau
            const headerRow = document.createElement('tr');
            Object.keys(data.data_preview[0]).forEach(column => {
                const th = document.createElement('th');
                th.textContent = column;
                headerRow.appendChild(th);
            });
            dataTable.appendChild(headerRow);

            // Ajouter les données au tableau
            data.data_preview.forEach(row => {
                const tr = document.createElement('tr');
                Object.values(row).forEach(cell => {
                    const td = document.createElement('td');
                    td.textContent = cell;
                    tr.appendChild(td);
                });
                dataTable.appendChild(tr);
            });

            // Afficher les tailles d'entraînement et de test
            document.getElementById('train_size').textContent = data.num_rows * 0.8;
            document.getElementById('test_size').textContent = data.num_rows * 0.2;
        }
    })
    .catch(error => console.error('Erreur:', error));
});



//js pour training models 






function openSummaryPage() {
    // Ouvrir la page summary.html dans un nouvel onglet
    window.open('summary.html', '_blank');
}












// Fonction pour mettre à jour la table des résultats
function updateResultsTable(data) {
    // Récupérer l'élément de la table
    const resultsTable = document.getElementById('results-table').getElementsByTagName('tbody')[0];
    
    // Vider la table existante
    resultsTable.innerHTML = '';
    
    // Parcourir les résultats reçus
    data.forEach(result => {
        // Créer une nouvelle ligne dans la table
        const row = document.createElement('tr');
        
        // Ajouter les cellules pour chaque donnée (classifier, accuracy, weight, F1-score)
        const modelCell = document.createElement('td');
        modelCell.textContent = result.classifier;
        row.appendChild(modelCell);
        
        const accuracyCell = document.createElement('td');
        accuracyCell.textContent = result.accuracy.toFixed(2);
        row.appendChild(accuracyCell);
        
        const weightCell = document.createElement('td');
        weightCell.textContent = result.weight ? result.weight.toFixed(2) : 'N/A';
        row.appendChild(weightCell);
        
        const f1Cell = document.createElement('td');
        f1Cell.textContent = result.f1_score.toFixed(2);
        row.appendChild(f1Cell);
        
        // Ajouter la ligne à la table
        resultsTable.appendChild(row);
    });
}

// Écouter l'événement de soumission du formulaire
document.getElementById('train-models-form').addEventListener('submit', function(event) {
    event.preventDefault(); // Empêcher la soumission par défaut du formulaire
    
    // Envoyer la requête POST à Flask
    const formData = new FormData(event.target);
    fetch('/train', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Appeler la fonction pour mettre à jour la table des résultats
        updateResultsTable(data);
    })
    .catch(error => {
        console.error('Erreur lors de la soumission du formulaire:', error);
    });
});













document.getElementById("upload-split-form").addEventListener("submit", function(event) {
    event.preventDefault();
    // Code to handle data upload and split
    // Call the training function after data upload and split
    trainModels();
});


function trainModels() {
    // Code to train models
    // Call the testing function after training is complete
    testModels();
}

function testModels() {
    // Code to test models
    // Call the prediction function after testing is complete
    predictAndVisualize();
}


function predictAndVisualize() {
    // Code to perform prediction and visualization
}
