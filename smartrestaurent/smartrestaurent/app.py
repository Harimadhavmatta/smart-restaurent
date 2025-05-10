from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import random
import string
from datetime import datetime
import pytz
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import spacy
import json
import re

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Define the sentence template
template = "As someone with [disease], I am looking for a meal that is rich in [wanted nutrients] while avoiding [allergic items] and featuring a [taste profile] flavor."


app = Flask(__name__)

def cluster_dishes(df, num_clusters=5, model_name='all-MiniLM-L6-v2', output_file="clustered_dishes.csv"):
    """
    Clusters dishes based on their information using SentenceTransformer embeddings and KMeans.

    Parameters:
    - df (pd.DataFrame): DataFrame containing dish information in a column named 'information'.
    - num_clusters (int): Number of clusters for KMeans.
    - model_name (str): Pretrained SentenceTransformer model name.
    - output_file (str): File path to save the clustered DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with an additional 'Cluster' column.
    """
    # Load the model
    model = SentenceTransformer(model_name)

    # Generate embeddings for dish information
    df['Joined_Information'] = df['information'].apply(', '.join)  # Combine information into a single string
    dish_embeddings = model.encode(df['Joined_Information'], convert_to_tensor=True)

    # Perform clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(dish_embeddings.cpu().numpy())

    # Display clustered dishes
    #print("Clustered Dishes:")
    for cluster_id, cluster_items in df.groupby('Cluster')['Joined_Information']:
        #print(f"Cluster {cluster_id}:")
        for item in cluster_items:
            print(f"  - {item}")

    # Reduce dimensionality for visualization
    df = df.drop('Joined_Information', axis=1)

    # Save the clustered DataFrame for further analysis
    df.to_csv('cluster.csv', index=False)
    
    
    
    
    
# Helper functions
def remove_and(input_string):
    pattern = r'[,\./\?]| and '
    result = re.split(pattern, input_string)
    result = [word.strip() for word in result if word.strip()]
    return result




# Function to replace sentences with the dominant word
def replace_sentence_with_dominant_word(sentences, left_patterns, right_patterns):
    if sentences is None:
        return []
    updated_sentences = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        pattern_found = False
        for pattern in left_patterns:
            if pattern in sentence_lower:
                parts = re.split(pattern, sentence_lower)
                if len(parts) > 1:
                    dominant_word = parts[0].strip().split()[-1]
                    updated_sentences.append(dominant_word)
                    pattern_found = True
                break
        if not pattern_found:
            for pattern in right_patterns:
                if pattern in sentence_lower:
                    parts = re.split(pattern, sentence_lower)
                    if len(parts) > 1:
                        dominant_word = parts[1].strip().split()[0]
                        updated_sentences.append(dominant_word)
                        pattern_found = True
                    break
        if not pattern_found:
            updated_sentences.append(sentence)
    return updated_sentences





# Function to compute similarity between a string and a list of strings
def compute_similarity(item, info_list):
    vectorizer = TfidfVectorizer().fit([item] + info_list)
    vectors = vectorizer.transform([item] + info_list)
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    return cosine_similarities




def filter_recommendations(df, final_wanted, final_not_wanted):
    
    try:
        print('hello there')
        
        df1_rows = []
        df2_rows = []

        # Convert wanted and not wanted items to lowercase
        final_wanted = [item.lower() for item in final_wanted]
        final_not_wanted = [item.lower() for item in final_not_wanted]

        # Ensure 'information' column is properly formatted as a list of lowercase items
        df['information'] = df['information'].astype(str).str.strip("[]").str.replace("'", "").str.split(", ", regex=True)
        df['information'] = df['information'].apply(lambda x: [i.strip().lower() for i in x] if isinstance(x, list) else str(x).lower().split(', '))

        # Filtering logic
        for _, row in df.iterrows():
            info_items = row['information']  # Extract list of food description items

            if isinstance(info_items, list):
                # Check if any unwanted item is in the row (substring match)
                if any(any(word in info_item for info_item in info_items) for word in final_not_wanted):
                    df2_rows.append(row)  # Append the full row
                # Otherwise, check if any wanted item is present
                if any(any(word in info_item for info_item in info_items) for word in final_wanted):
                    df1_rows.append(row)  # Append the full row

        # Convert to DataFrames
        df1 = pd.DataFrame(df1_rows, columns=df.columns)
        df2 = pd.DataFrame(df2_rows, columns=df.columns)

        # Ensure 'title' exists before filtering
        if 'title' in df1.columns and 'title' in df2.columns:
            df1_filtered = df1[~df1['title'].isin(df2['title'])]
            df1_filtered.to_csv('recommended.csv', index=False)
            return df1_filtered
        else:
            print("Columns missing!")
            return None

    except Exception as e:
        print("Error:", str(e))
        return None


# Function to extract values based on the template
def extract_meal_preferences(sentence):
    doc = nlp(sentence)
    print("Received input data:", doc)
    
    # Define regex patterns based on the template structure
    disease_pattern = r"As someone with (.*?), I am looking"
    nutrients_pattern = r"rich in (.*?) while avoiding"
    allergic_pattern = r"avoiding (.*?) and featuring"
    taste_pattern = r"featuring a (.*?) flavor"
    if not sentence:
        return [], [], [], []

    # Extract values using regex
    disease = re.findall(disease_pattern, sentence)
    nutrients = re.findall(nutrients_pattern, sentence)
    allergic_items = re.findall(allergic_pattern, sentence)
    taste_profile = re.findall(taste_pattern, sentence)
    print(taste_profile)

    # Convert extracted values to lists
    disease_list = disease[0].split(", ") if disease else []
    nutrients_list = nutrients[0].split(" and ") if nutrients else []
    allergic_list = allergic_items[0].split(" and ") if allergic_items else []
    taste_list = taste_profile[0].split(", ") if taste_profile else ["na"]

    return disease_list, nutrients_list, allergic_list, taste_list





# File paths
ORDERS_CSV_PATH = "orders.csv"
DATA_CSV_PATH = "data.csv"

# Helper function to generate a random order code
def generate_order_code(length=8):
    characters = string.ascii_uppercase + string.digits
    return ''.join(random.choices(characters, k=length))



# Route for home page
@app.route('/')
def home():
    return render_template('index.html')



@app.route('/caution')
def caution():
    print("Caution page is being served!")  # Debugging
    return render_template('caution.html')



# Route to display menu
@app.route('/menu')
def menu():
    try:
        df = pd.read_csv(DATA_CSV_PATH)
        selected_columns = ['ITEM_ID', 'title', 'ingredients', 'nutrition_info', 'price', 'country_of_origin']
        df = df[selected_columns]
        table_html = df.to_html(classes='custom-table', index=False, border=0)
        return render_template('table.html', table=table_html)
    except FileNotFoundError:
        return "Menu data file not found!", 500
    
    
  
    
    

# Route to display orders
@app.route('/orders')
def orders():
    try:
        df = pd.read_csv(ORDERS_CSV_PATH)
        orders_table_html = df.to_html(classes='custom-table', index=False, border=0)
        return render_template('orders.html', table=orders_table_html)
    except FileNotFoundError:
        return "Orders data file not found!", 500

    
    
# Route for staff page
@app.route('/staff')
def staff():
    return render_template('staff.html')




# Route for customer page
@app.route('/customer')
def customer():
    return render_template('customer.html')



@app.route('/add_dish', methods=["POST"])
def add_dish():
    try:
        # Get form data
        item_id = request.form.get('item-id')
        title = request.form.get('title')
        ingredients = request.form.get('ingredients')
        nutrition_info = request.form.get('nutrition-info')
        micros_macros = request.form.get('micros-macros')
        price = request.form.get('price')
        information = request.form.get('information')
        country_of_origin = request.form.get('country_of_origin')

        # Check if all fields are filled
        if not all([item_id, title, ingredients, nutrition_info, micros_macros, price, information, country_of_origin]):
            return jsonify({'error': 'All fields are required'}), 400

        # Try reading the CSV file
        try:
            df = pd.read_csv(DATA_CSV_PATH)
        except FileNotFoundError:
            df = pd.DataFrame(columns=["ITEM_ID", "title", "ingredients", "nutrition_info", "micros_and_macros", "price", "information", "country_of_origin"])

        # Check what the DataFrame looks like
        #print(f"DataFrame loaded: {df.head()}")

        new_entry = {
            "ITEM_ID": item_id,
            "title": title,
            "ingredients": ingredients,
            "nutrition_info": nutrition_info,
            "micros_and_macros": micros_macros,
            "price": price,
            "information": information,
            "country_of_origin": country_of_origin
        }

        # Debug: check the type of new_entry
        #print(f"new_entry: {new_entry}")

        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        
        # Debug: Check DataFrame before saving
        #print(f"Updated DataFrame: {df.head()}")

        df.to_csv(DATA_CSV_PATH, index=False)

        # Check what 'cluster_dishes' does
        data=pd.read_csv("data.csv")
        cluster_dishes(data)

        return jsonify({'message': 'Dish added successfully'}), 200

    except Exception as e:
        # Log the error for better insight
        #print(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
    
    
    

# Route to remove a dish
@app.route('/remove_dish', methods=['POST'])
def remove_dish():
    item_id = request.form.get('remove-item-id')

    if not item_id:
        return jsonify({'error': 'Item ID is required'}), 400

    try:
        df = pd.read_csv(DATA_CSV_PATH)
        if item_id not in df['ITEM_ID'].astype(str).values:
            return jsonify({'error': f'Item ID {item_id} not found'}), 404

        df = df[df['ITEM_ID'].astype(str) != item_id]
        df.to_csv(DATA_CSV_PATH, index=False)

        return jsonify({'message': f'Item ID {item_id} removed successfully'}), 200

    except FileNotFoundError:
        return jsonify({'error': 'Menu file not found'}), 500

    
    #========================================scrap
@app.route('/context_recommend', methods=['POST'])
def context_recommend():
    try:
        # Read the recommended.csv file
        df = pd.read_csv('recommended.csv')

        # Convert DataFrame to list of dictionaries for rendering
        table_data = df.to_dict(orient='records')
        print('yess')
        # Render the table.html template with data
        return render_template('table.html', table_data=table_data)
    
    except Exception as e:
        return f"Error loading recommendations: {str(e)}"
    
    #=========================================================
    
# Route to place an order
@app.route('/place_order', methods=['POST'])
def place_order():
    try:
        menu = pd.read_csv(DATA_CSV_PATH)
        order_id = generate_order_code()
        item_id = request.form.get('order-item-id')
        table_number = request.form.get('table-number')
        quantity = request.form.get('quantity')

        local_timezone = pytz.timezone("Asia/Kolkata")
        current_datetime = datetime.now(local_timezone)
        date = str(current_datetime.date())
        time = str(current_datetime.time())
        status = "in-progress"

        if item_id not in menu['ITEM_ID'].values:
            return jsonify({'error': f'Item ID {item_id} not found in menu'}), 404

        price = float(menu[menu['ITEM_ID'] == item_id]['price'].values[0])
        bill = int(quantity) * price

        try:
            orders = pd.read_csv(ORDERS_CSV_PATH)
        except FileNotFoundError:
            orders = pd.DataFrame(columns=["order_id", "ITEM_ID", "table_number", "Quantity", "date", "time", "status", "bill"])

        new_order = {
            "order_id": order_id,
            "ITEM_ID": item_id,
            "table_number": table_number,
            "Quantity": int(quantity),
            "date": date,
            "time": time,
            "status": status,
            "bill": bill
        }

        orders = pd.concat([orders, pd.DataFrame([new_order])], ignore_index=True)
        orders.to_csv(ORDERS_CSV_PATH, index=False)

        return jsonify({'message': 'Order placed successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
    
    
# Route to display the bill
@app.route("/billing", methods=["POST"])
def billing():
    try:
        orders = pd.read_csv(ORDERS_CSV_PATH)
        table_number = request.form.get("Table_number")
        tip = float(request.form.get("tip", 0))

        table_orders = orders[orders["table_number"] == table_number]
        if table_orders.empty:
            return render_template("bill.html", bill_html="<h3>No orders found for the given table number.</h3>")

        total_bill = table_orders["bill"].sum()
        total_with_tip = total_bill + tip

        bill_html = f"""
        <h3>Bill for Table {table_number}</h3>
        <table border="1" style="border-collapse: collapse;">
            <tr>
                <th>Order ID</th>
                <th>Item ID</th>
                <th>Quantity</th>
                <th>Bill</th>
            </tr>
        """

        for _, order in table_orders.iterrows():
            bill_html += f"""
            <tr>
                <td>{order['order_id']}</td>
                <td>{order['ITEM_ID']}</td>
                <td>{order['Quantity']}</td>
                <td>{order['bill']}</td>
            </tr>
            """

        bill_html += f"""
            <tr>
                <td colspan="3" style="text-align:right;">Total</td>
                <td>{total_bill}</td>
            </tr>
            <tr>
                <td colspan="3" style="text-align:right;">Tip</td>
                <td>{tip}</td>
            </tr>
            <tr>
                <td colspan="3" style="text-align:right;">Total with Tip</td>
                <td>{total_with_tip}</td>
            </tr>
        </table>
        """

        return render_template("bill.html", bill_html=bill_html)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    
    

@app.route('/filtered_recommendations', methods=['POST'])
def filtered_recommendations():
    # Get the JSON data from the request
    data = request.get_json()
    df = pd.read_csv('data.csv')
    # Extract wanted and not wanted ingredients
    wanted_ingredients = data.get('wanted', [])
    not_wanted_ingredients = data.get('not_wanted', [])
#     print(len(wanted_ingredients))
#     print(len(not_wanted_ingredients))
#     print(type(wanted_ingredients))
#     print(type(not_wanted_ingredients))
    df1_filtered = filter_recommendations(df,wanted_ingredients, not_wanted_ingredients)
    recommendations_list = df1_filtered.to_dict(orient='records')
    # Here you would implement your recommendation logic
    # For example, query a database for recipes that match the criteria
    
    # For this example, we'll just return the data back
    recommendations = {
        "success": True,
        "wanted": wanted_ingredients,
        "not_wanted": not_wanted_ingredients,
        "recommendations": recommendations_list 
    }
    print('hello')
    
    return jsonify(recommendations)

@app.route('/recommended')
def recommended_page():
    # This is the page that will display the recommendations
    # The actual data will be loaded from localStorage on the client side
    return render_template('recommended.html')



@app.route('/process', methods=['POST'])
def process_context():
    try:
        # Load input context from the request
        input_data = request.get_json()
        context = input_data.get('context')
        if not context:
            return jsonify({'error': 'No context provided'}), 400


        # Extract Information (Ensure this function is defined elsewhere)
        disease, nutrients, allergic_items, taste_profile = extract_meal_preferences(context)

        # Debugging prints
        print("Disease:", disease)
        print("Wanted Nutrients:", nutrients)
        print("Allergic Items:", allergic_items)
        print("Taste Profile:", taste_profile)

        # Load the disease dataset
        disease_df = pd.read_csv('custom.csv')
        disease_list = disease
        wanted_items = set()
        not_wanted_items = set()

        # Populate wanted_items and not_wanted_items based on disease data
        for disease in disease_list:
            matched_rows = disease_df[disease_df['name'].str.lower() == disease.lower()]
        
            for _, row in matched_rows.iterrows():
                wanted_items.update(map(str.strip, row['wanted'].split(',')))
                not_wanted_items.update(map(str.strip, row['not wanted'].split(',')))

        print("Wanted Items:", wanted_items)
        print("Not Wanted Items:", not_wanted_items)

        # Remove common items from wanted_items and not_wanted_items
        common_items = wanted_items.intersection(not_wanted_items)
        wanted_items -= common_items
        final_wanted = list(wanted_items) + taste_profile
        final_not_wanted = list(not_wanted_items) + allergic_items

        print('Final Wanted:', final_wanted)
        print('Final Not Wanted:', final_not_wanted)
        preferences = {
            "wanted": final_wanted,
            "notWanted": final_not_wanted
        }

        # Save to JSON file
        with open("static/preferences.json", "w") as file:
            json.dump(preferences, file, indent=2)

        # Load the main dataset
        df = pd.read_csv('data.csv')

        # Clean 'information' column
        df['information'] = df['information'].astype(str).str.strip("[]").str.replace("'", "").str.split(", ", regex=True)

        # Define column structure
        columns = ['ITEM_ID', 'title', 'ingredients', 'nutrition_info', 'micros_and_macros', 'price', 'information', 'country_of_origin']

        # Initialize empty DataFrames
        df1_rows = []
        df2_rows = []

        # Ensure 'information' is properly formatted as a list
        df['information'] = df['information'].apply(lambda x: [i.strip().lower() for i in x] if isinstance(x, list) else str(x).lower().split(', '))

        # Convert final_wanted and final_not_wanted to lowercase for case-insensitive matching
        final_wanted = [item.lower() for item in final_wanted]
        final_not_wanted = [item.lower() for item in final_not_wanted]
        count=0
        # Filter rows based on wanted and not wanted items
        for _, row in df.iterrows():
            info_items = row['information']  # Extract list of food description items
#             print('context',info_items)
            count+=1
            if isinstance(info_items, list):
                # Check if any unwanted item is in the row (substring match)
                if any(any(word in info_item for info_item in info_items) for word in final_not_wanted):
#                     print("yes context")
                    df2_rows.append(row)  # Append the full row
                # Otherwise, check if any wanted item is present
                if any(any(word in info_item for info_item in info_items) for word in final_wanted):
#                     print("no context")
                    
                    df1_rows.append(row)  # Append the full row
        print('contect iter ',count)
        # Convert to DataFrames
        df1 = pd.DataFrame(df1_rows, columns=df.columns)
        df2 = pd.DataFrame(df2_rows, columns=df.columns)
        

        # Debugging prints
        print("df1 Columns:", df1.columns.tolist())
        print("df2 Columns:", df2.columns.tolist())
        print("df1 Length:", len(df1))
        print("df2 Length:", len(df2))
        
        # Ensure 'title' exists in both DataFrames before filtering
        if 'title' in df1.columns and 'title' in df2.columns:
            df1_filtered = df1[~df1['title'].isin(df2['title'])]
            n = len(df1_filtered)
            print("Filtered DataFrame Length:", len(df1_filtered))

            # Save the filtered DataFrame to a CSV file
            df1_filtered.to_csv('recommended.csv', index=False)
            message = f"You have {n} recommendations on the recommendations page if you are not satisfied with the current recommendations."
            return render_template("table.html", message=message)
        else:
            print("Columns missing!")
            return redirect(url_for('caution'))

        # Print the filtered DataFrame length
        print("Filtered DataFrame Length:", len(df1_filtered))

        # Save the filtered DataFrame to a CSV file
        df1_filtered.to_csv('recommended.csv', index=False)

        # Return the result as JSON
        return jsonify(df1_filtered.to_dict(orient='records'))

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500







# Run the app
if __name__ == '__main__':
    app.run()
