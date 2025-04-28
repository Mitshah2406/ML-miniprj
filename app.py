from flask import Flask, render_template, request
import joblib
import pandas as pd
import io
import base64
import os


import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns


app = Flask(__name__)


model = joblib.load("traffic_model.pkl") 


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    hour = int(request.form['hour'])
    junction = int(request.form['junction'])
    day = int(request.form['day'])
    month = int(request.form['month'])


    data = pd.DataFrame({'Junction': [junction],
                         'Month': [month],
                         'Day': [day],
                         'Hour': [hour]})


    prediction = model.predict(data)


    all_junctions_traffic = {}
    for j in range(1, 5):
        junction_data = pd.DataFrame({'Junction': [j],
                                     'Month': [month],
                                     'Day': [day],
                                     'Hour': [hour]})
        junction_prediction = model.predict(junction_data)
        all_junctions_traffic[j] = round(junction_prediction[0])
    

    best_junction = min(all_junctions_traffic, key=all_junctions_traffic.get)
    

    junctions = list(all_junctions_traffic.keys())
    traffic_values = list(all_junctions_traffic.values())
    

    plt.figure(figsize=(10, 6))
    plt.clf()  
    bars = plt.bar(junctions, traffic_values, color=['blue', 'green', 'orange', 'red'])
    plt.title(f'Predicted Traffic by Junction (Month: {month}, Day: {day}, Hour: {hour})')
    plt.xlabel('Junction')
    plt.ylabel('Number of Vehicles')
    plt.xticks(junctions)
    

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom')
    

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()  
    

    graph = base64.b64encode(image_png).decode('utf-8')
    

    return render_template('result.html', 
                          requested_junction=junction,
                          prediction=round(prediction[0]), 
                          hour=hour, 
                          day=day, 
                          month=month, 
                          all_junctions=all_junctions_traffic,
                          best_junction=best_junction, 
                          graph=graph)


@app.route('/visualize')
def visualize():
    try:

        df = pd.read_csv('dataset/traffic.csv')
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['Hour'] = df['DateTime'].dt.hour
        

        plt.figure(figsize=(12, 6))
        plt.clf()  
        hourly_avg = df.groupby('Hour')['Vehicles'].mean()
        sns.lineplot(x=hourly_avg.index, y=hourly_avg.values)
        plt.title('Average Hourly Traffic')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Number of Vehicles')
        plt.xticks(range(0, 24))
        plt.grid(True)
        

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()  
        

        hourly_graph = base64.b64encode(image_png).decode('utf-8')
        

        plt.figure(figsize=(10, 6))
        plt.clf() 
        junction_avg = df.groupby('Junction')['Vehicles'].mean()
        sns.barplot(x=junction_avg.index, y=junction_avg.values)
        plt.title('Average Traffic by Junction')
        plt.xlabel('Junction')
        plt.ylabel('Average Number of Vehicles')
        

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()  
        

        junction_graph = base64.b64encode(image_png).decode('utf-8')
        
        return render_template('visualize.html', hourly_graph=hourly_graph, junction_graph=junction_graph)
    except Exception as e:
        return render_template('visualize.html', error=str(e))

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True, host='0.0.0.0')