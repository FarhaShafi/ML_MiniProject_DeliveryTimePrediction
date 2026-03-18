from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl','rb'))
print('Model Loaded')

Item_Name_en = pickle.load(open('Item_Name.pkl','rb'))
City_en=pickle.load(open('City.pkl','rb'))
Driver_Vehicle_en=pickle.load(open('Driver_Vehicle.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML page

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        try:
            # Get form inputs
            Item_Name = request.form['Item_Name']
            print(Item_Name)
            Quantity = int(request.form['Quantity'])
            print(Quantity)
            Total_Price = float(request.form['Total_Price'])
            print(Total_Price)
            City = request.form['City']
            print(City)
            Payment_Method = int(request.form['Payment_Method'])
            print(Payment_Method)
            Order_Status = int(request.form['Order_Status'])
            print(Order_Status)
            Driver_Vehicle = int(request.form['Driver_Vehicle'])
            print(Driver_Vehicle)
            Delivery_Distance_km = int(request.form['Delivery_Distance_km'])
            print(Delivery_Distance_km)
            Traffic_Level = int(request.form['Traffic_Level'])
            print(Traffic_Level)
            Driver_Availability_Online = int(request.form['Driver_Availability'])
            print(Driver_Availability_Online)

            Item_Name_val = Item_Name_en.transform([Item_Name])[0]
            print(Item_Name_val)

            City_val = City_en.transform([City])[0]
            print(City_val)

            # Prepare data 
            details =[Item_Name_val, Quantity, Total_Price, City_val, Payment_Method, Order_Status, Driver_Vehicle, Delivery_Distance_km, Traffic_Level, Driver_Availability_Online]
            print(details)

            data_out=np.array(details).reshape(1,-1)
            print(data_out)
            print(data_out.shape)

            scaled = pickle.load(open('scaling.pkl','rb'))
            data_scaled = scaled.transform(data_out)

            # Predict Food Delivery Duration in Minutes
            prediction = model.predict(data_scaled)
            print(prediction)

            return render_template('index.html', prediction_text=f'Estimated Food Delivery Time : {float(round(prediction[0],2))} minutes')
        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
