# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Streamlit app implementation
def main():
    st.title("Soil Quality Classification System")
    st.write("Upload soil sample data to classify soil types and analyze soil characteristics")
    
    # File upload section
    uploaded_file = st.file_uploader("Choose a CSV file with soil data", type="csv")
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.success("Data uploaded successfully!")
        
        # Show raw data
        if st.checkbox("Show raw data"):
            st.write(df)
        
        # Data preprocessing
        st.header("Data Preprocessing")
        
        # Select features and target
        st.subheader("Select columns for analysis")
        all_columns = df.columns.tolist()
        
        # Let user select target variable
        target_column = st.selectbox("Select target column (soil type/class):", all_columns)
        
        # Let user select feature columns
        potential_features = [col for col in all_columns if col != target_column]
        selected_features = st.multiselect(
            "Select feature columns:", 
            potential_features,
            default=[col for col in potential_features if any(x in col.lower() for x in ['ph', 'nutrient', 'moisture', 'nitrogen', 'phosphorus', 'potassium'])]
        )
        
        if len(selected_features) > 0 and target_column:
            # Prepare data
            X = df[selected_features]
            y = df[target_column]
            
            # Encode target if it's categorical
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
                class_names = le.classes_
            else:
                class_names = sorted(y.unique())
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train-test split
            test_size = st.slider("Test set size:", 0.1, 0.5, 0.2)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
            
            st.write(f"Training set: {X_train.shape[0]} samples")
            st.write(f"Test set: {X_test.shape[0]} samples")
            
            # Model selection
            st.header("Model Selection and Training")
            model_option = st.selectbox(
                "Choose a machine learning model:",
                ["Decision Tree", "Support Vector Machine (SVM)", "Neural Network"]
            )
            
            # Train selected model
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    if model_option == "Decision Tree":
                        model = DecisionTreeClassifier(random_state=42)
                    elif model_option == "Support Vector Machine (SVM)":
                        model = SVC(probability=True, random_state=42)
                    else:  # Neural Network
                        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
                    
                    # Fit the model
                    model.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred = model.predict(X_test)
                    
                    # Evaluation
                    st.header("Model Evaluation")
                    
                    # Classification report
                    st.subheader("Classification Report")
                    report = classification_report(y_test, y_pred, target_names=[str(c) for c in class_names], output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.write(report_df)
                    
                    # F1 Score
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    st.metric("F1 Score (Weighted)", f"{f1:.4f}")
                    
                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=class_names, yticklabels=class_names)
                    plt.ylabel('Actual')
                    plt.xlabel('Predicted')
                    st.pyplot(fig)
                    
                    # Feature importance (for Decision Tree)
                    if model_option == "Decision Tree":
                        st.subheader("Feature Importance")
                        feature_imp = pd.DataFrame({
                            'Feature': selected_features,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(feature_imp, x='Importance', y='Feature', orientation='h',
                                    title='Feature Importance for Soil Classification')
                        st.plotly_chart(fig)
            
            # Data visualization
            st.header("Data Visualization")
            
            # Correlation heatmap
            st.subheader("Correlation Heatmap")
            corr = df[selected_features].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
            st.pyplot(fig)
            
            # Distribution of soil types
            st.subheader("Distribution of Soil Types")
            soil_counts = df[target_column].value_counts().reset_index()
            soil_counts.columns = ['Soil Type', 'Count']
            fig = px.pie(soil_counts, values='Count', names='Soil Type', title='Soil Type Distribution')
            st.plotly_chart(fig)
            
            # Scatter plots for selected features
            st.subheader("Feature Relationships")
            if len(selected_features) >= 2:
                x_axis = st.selectbox("X-axis", selected_features)
                y_axis = st.selectbox("Y-axis", [f for f in selected_features if f != x_axis])
                
                fig = px.scatter(df, x=x_axis, y=y_axis, color=target_column,
                               title=f'{y_axis} vs {x_axis} by Soil Type',
                               labels={x_axis: x_axis, y_axis: y_axis})
                st.plotly_chart(fig)
            
            # Regional analysis if geographic data available
            if any(col.lower() in ['region', 'location', 'area', 'zone', 'lat', 'lon', 'latitude', 'longitude'] for col in df.columns):
                st.subheader("Regional Soil Analysis")
                region_col = [col for col in df.columns if col.lower() in ['region', 'location', 'area', 'zone']][0] if any(col.lower() in ['region', 'location', 'area', 'zone'] for col in df.columns) else None
                
                if region_col:
                    # Average soil characteristics by region
                    region_data = df.groupby(region_col)[selected_features].mean().reset_index()
                    
                    # Let user select which characteristic to view
                    feature_to_view = st.selectbox("Select soil characteristic:", selected_features)
                    
                    fig = px.bar(region_data, x=region_col, y=feature_to_view,
                               title=f'Average {feature_to_view} by Region')
                    st.plotly_chart(fig)
            
            # Optimal soil conditions for crops section
            st.header("Optimal Soil Conditions")
            st.write("Identify optimal soil conditions for different crops based on classification results")
            
            if st.checkbox("Show optimal soil conditions analysis"):
                # This would typically connect to a database of crop requirements
                # For this example, we'll create a simple placeholder analysis
                st.info("This feature would connect to crop requirement data to provide detailed recommendations.")
                
                # Example visualization
                crops = ['Wheat', 'Corn', 'Rice', 'Soybeans', 'Cotton']
                ph_requirements = [6.5, 6.0, 5.5, 6.8, 6.2]
                nitrogen_needs = [70, 90, 100, 60, 85]
                
                crop_data = pd.DataFrame({
                    'Crop': crops,
                    'Optimal pH': ph_requirements,
                    'Nitrogen Need (kg/ha)': nitrogen_needs
                })
                
                st.write(crop_data)
                
                fig = go.Figure()
                for idx, crop in enumerate(crops):
                    fig.add_trace(go.Bar(
                        x=[crop],
                        y=[nitrogen_needs[idx]],
                        name=crop
                    ))
                
                fig.update_layout(title='Nitrogen Requirements by Crop', 
                                yaxis_title='Nitrogen Need (kg/ha)')
                st.plotly_chart(fig)
                
        else:
            st.warning("Please select at least one feature column and a target column.")
    else:
        # Show sample data and instructions when no file is uploaded
        st.info("Please upload a CSV file with soil sample data to begin analysis.")
        st.write("Your data should include columns for:")
        st.write("- Soil nutrients (N, P, K, etc.)")
        st.write("- pH levels")
        st.write("- Moisture content")
        st.write("- Soil classification/type (target variable)")
        
        # Show sample format
        st.subheader("Sample Data Format:")
        sample_data = {
            'pH': [6.5, 7.2, 5.8, 6.9, 5.5],
            'Nitrogen': [45, 32, 60, 38, 55],
            'Phosphorus': [20, 15, 25, 18, 22],
            'Potassium': [80, 65, 90, 75, 85],
            'Moisture': [25, 18, 30, 22, 28],
            'SoilType': ['Clay', 'Sandy', 'Loam', 'Clay', 'Loam']
        }
        st.write(pd.DataFrame(sample_data))

if __name__ == "__main__":
    main()
    