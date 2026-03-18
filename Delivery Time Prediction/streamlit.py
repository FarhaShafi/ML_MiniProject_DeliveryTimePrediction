import streamlit as st
import pickle
import numpy as np

# Load trained model and encoders
model = pickle.load(open('model.pkl', 'rb'))
Item_Name_en = pickle.load(open('Item_Name.pkl','rb'))
City_en=pickle.load(open('City.pkl','rb'))
Driver_Vehicle_en=pickle.load(open('Driver_Vehicle.pkl','rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

st.set_page_config(page_title="Food Delivery Time Prediction")
st.title("Food Delivery Time Prediction App")
st.write("Enter the order details below to estimate its delivery time:")

# --- Input fields ---
col1, col2 = st.columns(2)

with col1:
    Item_Name = st.text_input("Selected Item (e.g. Fried Chicken)", "Fried Chicken")
    Quantity = st.text_input("Quantity", "2")
    Total_Price = st.text_input("Total Price", "100")
    City = st.text_input("City", "Alexandria")
    Delivery_Distance_km = st.text_input("Delivery Distance (In kms)", "5")

with col2:
    Payment_Method = st.selectbox("Payment Method", ("Wallet", "Credit Card", "Cash"))
    Order_Status = st.selectbox("Order Status", ("Delivered", "In Transit", "Cancelled"))
    Driver_Vehicle = st.selectbox("Driver Vehicle", ("Bicycle", "Car", "Motorbike"))
    Traffic_Level = st.selectbox("Traffic Level", ("High", "Low", "Medium"))
    Driver_Availability_Online = st.selectbox("Driver Availability", ("Offline", "Online"))

# --- Input conversion and validation ---
try:
    Quantity = int(Quantity)
    Total_Price = float(Total_Price)
    Delivery_Distance_km = float(Delivery_Distance_km)
    Order_Status = int(Order_Status)
    Payment_Method = int(Payment_Method)
    Driver_Vehicle = int(Driver_Vehicle)
    Traffic_Level = int(Traffic_Level)

    # Encode categorical variables
    Item_Name_val = Item_Name_en.transform([Item_Name])[0]
    City_val = City_en.transform([City])[0]
    Driver_Vehicle_val = Driver_Vehicle_en.transform([Driver_Vehicle])[0]
    Driver_Availability_Online_val = 1 if Driver_Availability_Online == "Online" else 0

    # Prepare input data
    details =[int(Item_Name), Quantity, Total_Price, City, Payment_Method, Order_Status, Driver_Vehicle, Delivery_Distance_km, Traffic_Level, Driver_Availability]
    data_out = np.array(details).reshape(1, -1)
    data_scaled = scaler.transform(data_out)

    if st.button("Predict Food Delivery Time"):
        prediction = model.predict(data_scaled)[0]
        st.success(f"Estimated Food Delivery Time:{round(prediction, 2)}")

except ValueError:
    st.warning("Please enter valid numeric values for Quantity, Total Price and Delivery Distance.")
