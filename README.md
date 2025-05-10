# smart-restaurent
🍽️ Smart Restaurant Recommendation System
This is a Smart Restaurant web application developed using Flask that offers personalized dish recommendations based on context using Natural Language Processing (NLP). It supports functionalities for customers, staff, and administrators, including dish management and real-time recommendations.

🚀 Features
🏠 Home Page – Welcome page with options to explore.

📜 Menu Page – Displays a list of dishes with clustering-based recommendations.

🧑‍🍳 Staff Page – Add, remove, or update dishes.

👥 Customer Page – Personalized recommendations based on input (e.g., "I feel cold" → hot soup).

🗂️ Order Page – Track your orders.

🔍 Context-based NLP Recommendations – Uses spaCy for named entity recognition and SentenceTransformer + KMeans for dish clustering.

🧠 Technologies Used
Python (Flask)

NLP: spaCy, SentenceTransformer

Machine Learning: KMeans Clustering

Web: HTML, CSS, Bootstrap

Data: JSON-based menu storage (or extend to SQL/NoSQL)

