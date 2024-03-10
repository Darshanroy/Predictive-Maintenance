import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import logging

class DataProcessing:
    def __init__(self, data):
        self.raw_data = data
        self.processed_data_for_target = None

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def preprocess_data_for_target(self):
        self.raw_data['Type'] = self.raw_data['Type'].map({'H': 2, 'M': 1, 'L': 0})
        # Perform One-Hot Encoding
        one_hot_encoded = pd.get_dummies(self.raw_data['Failure Type'])
        # Concatenate the encoded DataFrame with the original DataFrame
        raw_data_encoded = pd.concat([self.raw_data.drop(columns='Failure Type'), one_hot_encoded], axis=1)
        self.processed_data_for_target = raw_data_encoded.drop(['UDI', 'Product ID'], axis=1)
        print(self.processed_data_for_target.columns)

        target_variable = 'Target'
        x = self.processed_data_for_target.drop(target_variable, axis=1)
        x.columns = ['Type', 'Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min', 'Heat_Dissipation_Failure', 'No_Failure', 'Overstrain_Failure', 'Power_Failure', 'Random_Failures', 'Tool_Wear_Failure']
        y = self.processed_data_for_target[target_variable]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Initialize SMOTE
        smote = SMOTE(random_state=42)

        # Perform oversampling with SMOTE
        self.x_train, self.y_train = smote.fit_resample(self.x_train, self.y_train)


        return self.x_train, self.x_test, self.y_train, self.y_test

