import numpy as np                                                                                               
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder                                                    
from sklearn.model_selection import train_test_split                                                             
from sklearn.impute import SimpleImputer                                                                         
from sklearn.decomposition import PCA                                                                            
                                                                                                                 
def clean_data(data_points, labels):                                                                             
    # Step 1: Remove any invalid or missing data points                                                          
    valid_indices = np.all(np.isfinite(data_points), axis=1)                                                     
    data_points = data_points[valid_indices]                                                                     
    labels = labels[valid_indices]                                                                               
                                                                                                                 
    return data_points, labels                                                                                   
                                                                                                                 
def preprocess_data(data_file, labels_file, n_components, random_state):                                         
    # Step 1: Load the data                                                                                      
    data_points = np.loadtxt(data_file, delimiter=',')                                                           
    labels = np.loadtxt(labels_file)                                                                             
                                                                                                                 
    # Step 2: Clean the data                                                                                     
    data_points, labels = clean_data(data_points, labels)                                                        
                                                                                                                 
    # Step 3: Data pre-processing                                                                                
    # Normalize the first 64 values                                                                              
    minmax_scaler = MinMaxScaler()                                                                               
    data_points[:, :64] = minmax_scaler.fit_transform(data_points[:, :64])                                       
                                                                                                                 
    # One-hot encode the final 7 values                                                                          
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')                                              
    onehot_encoded = onehot_encoder.fit_transform(data_points[:, -7:])                                           
    data_points = np.concatenate((data_points[:, :-7], onehot_encoded), axis=1)                                  
                                                                                                                 
    # Handle missing data with mean imputation                                                                   
    imputer = SimpleImputer(strategy='mean')                                                                     
    data_points = imputer.fit_transform(data_points)                                                             
                                                                                                                 
    # Perform feature scaling/normalization                                                                      
    scaler = MinMaxScaler()                                                                                      
    data_points = scaler.fit_transform(data_points)                                                              
                                                                                                                 
    # Dimensionality reduction with PCA                                                                          
    pca = PCA(n_components)  # Reduce to n_components principal components                                       
    data_points = pca.fit_transform(data_points)                                                                 
                                                                                                                 
    # Step 4: Split the dataset into training and validation subsets                                             
    X_train, X_val, y_train, y_val = train_test_split(data_points, labels, test_size=0.2, random_state=random_state)
                                                                                                                 
    return X_train, X_val, y_train, y_val                                                                        
                                                                                                                 
# Example usage                                                                                                  
n_components = 6                                                                                                 
random_state = 10                                                                                                
data_file = 'traindata.txt'                                                                                      
labels_file = 'trainlabels.txt'                                                                                  
X_train, X_val, y_train, y_val = preprocess_data(data_file, labels_file, n_components, random_state)             
#print(X_val)                                                                                                    
                                                                                                                 
