import streamlit as st
import pandas as pd
import joblib


model = joblib.load("churn_rf_model.pkl")
kmeans = joblib.load("customer_segment_model.pkl")
scaler = joblib.load("scaler.pkl")

le_category = joblib.load("le_category.pkl")
le_payment = joblib.load("le_payment.pkl")
le_gender = joblib.load("le_gender.pkl")


features_to_scale = [
    'Total Purchase Amount',
    'Quantity',
    'Returns',
    'Age',
    'Product Price'
]

# IMPORTANT: Cluster must be included
feature_order_model = [
    'Product Category',
    'Product Price',
    'Quantity',
    'Total Purchase Amount',
    'Payment Method',
    'Returns',
    'Age',
    'Gender',
    'Cluster'
]

segment_dict = {
    0: "Premium Risk Customers",
    1: "High Spending Customers",
    2: "Low Value Customers",
    3: "Frequent Return Customers"
}


st.set_page_config(page_title="Customer Segmentation & Churn System", layout="wide")

st.title("Customer Segmentation & Churn Prediction System")

menu = ["Single Customer", "Batch Upload", "Dashboard"]
choice = st.sidebar.selectbox("Select Option", menu)


if choice == "Single Customer":

    st.subheader("Enter Customer Details")

    customer_age = st.number_input("Customer Age", 10, 100, 30)
    gender = st.selectbox("Gender", le_gender.classes_)
    product_category = st.selectbox("Product Category", le_category.classes_)
    payment_method = st.selectbox("Payment Method", le_payment.classes_)
    product_price = st.number_input("Product Price", 0, 10000, 50)
    quantity = st.number_input("Quantity Purchased", 1, 20, 1)
    
    # Automatically calculate total purchase
    total_purchase = product_price * quantity
    st.write(f"**Total Purchase Amount:** {total_purchase}")

    returns = st.number_input("Number of Returns", 0, 10, 0)

    if st.button("Predict Churn & Segment"):

        # Encode categorical
        category_enc = le_category.transform([product_category])[0]
        payment_enc = le_payment.transform([payment_method])[0]
        gender_enc = le_gender.transform([gender])[0]

        # Create dataframe
        new_customer = pd.DataFrame([{
            'Product Category': category_enc,
            'Product Price': product_price,
            'Quantity': quantity,
            'Total Purchase Amount': total_purchase,  # auto-calculated
            'Payment Method': payment_enc,
            'Returns': returns,
            'Age': customer_age,
            'Gender': gender_enc
        }])

        # Scale numeric features
        new_customer[features_to_scale] = scaler.transform(new_customer[features_to_scale])

        # -----------------------------
        # Predict Segment
        # -----------------------------
        cluster_features = new_customer[['Product Price','Quantity','Total Purchase Amount','Returns']]
        cluster = kmeans.predict(cluster_features)[0]

        new_customer["Cluster"] = cluster

        segment_name = segment_dict.get(cluster, "Unknown")

        # -----------------------------
        # Predict Churn
        # -----------------------------
        new_customer = new_customer[feature_order_model]

        churn_pred = model.predict(new_customer)[0]
        churn_prob = model.predict_proba(new_customer)[0][1]

        # -----------------------------
        # Show Results
        # -----------------------------
        st.subheader("Prediction Results")

        st.write(f"Customer Segment: **{segment_name}**")

        if churn_pred == 1:
            st.error("Customer is likely to CHURN")
        else:
            st.success("Customer is NOT likely to churn")

        st.write(f"Churn Probability: **{churn_prob*100:.2f}%**")

        if churn_prob > 0.7:
            st.warning("⚠️ High risk of churn!")

        

elif choice == "Batch Upload":

    st.header("Upload Customer CSV for Batch Prediction")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        st.write("Dataset Preview")
        st.dataframe(df.head())

        # Encode categorical
        df['Gender'] = le_gender.transform(df['Gender'])
        df['Product Category'] = le_category.transform(df['Product Category'])
        df['Payment Method'] = le_payment.transform(df['Payment Method'])

        # Scale numeric
        df[features_to_scale] = scaler.transform(df[features_to_scale])

        # Predict cluster
        cluster_features = df[['Product Price','Quantity','Total Purchase Amount','Returns']]
        df['Cluster'] = kmeans.predict(cluster_features)

        # Map segment names
        df['Segment'] = df['Cluster'].map(segment_dict)

        # Predict churn
        df_model = df[feature_order_model]

        df['Churn_Pred'] = model.predict(df_model)
        df['Churn_Prob'] = model.predict_proba(df_model)[:,1]

        st.subheader("Batch Prediction Results")
        st.dataframe(df)

        st.download_button(
            "Download Predictions",
            df.to_csv(index=False),
            file_name="customer_predictions.csv"
        )

elif choice == "Dashboard":

    st.header("Customer Segmentation Dashboard")

    uploaded_file = st.file_uploader("Upload CSV for Dashboard", type=["csv"])

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        df['Gender'] = le_gender.transform(df['Gender'])
        df['Product Category'] = le_category.transform(df['Product Category'])
        df['Payment Method'] = le_payment.transform(df['Payment Method'])

        df[features_to_scale] = scaler.transform(df[features_to_scale])

        cluster_features = df[['Product Price','Quantity','Total Purchase Amount','Returns']]
        df['Cluster'] = kmeans.predict(cluster_features)

        df['Segment'] = df['Cluster'].map(segment_dict)

        df_model = df[feature_order_model]

        df['Churn_Pred'] = model.predict(df_model)
        df['Churn_Prob'] = model.predict_proba(df_model)[:,1]

        st.subheader("Customer Segment Distribution")
        st.bar_chart(df['Segment'].value_counts())

        st.subheader("High Risk Customers")
        high_risk = df[df['Churn_Prob'] > 0.7]

        st.dataframe(high_risk)