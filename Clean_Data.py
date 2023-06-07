import numpy as np                                                                                          
from sklearn.preprocessing import StandardScaler                                                            
from sklearn.model_selection import train_test_split                                                        
from sklearn.impute import SimpleImputer                                                                    
from sklearn.decomposition import PCA                                                                       
                                                                                                            
def clean_data(data_points, labels):                                                                        
    # Step 1: Remove any invalid or missing data points                                                     
    valid_indices = np.all(np.isfinite(data_points), axis=1)                                                
    data_points = data_points[valid_indices]                                                                
    labels = labels[valid_indices]                                                                          
                                                                                                            
    # Step 2: Remove outliers using Z-score method                                                          
    z_scores = np.abs((data_points - np.mean(data_points, axis=0)) / np.std(data_points, axis=0))           
    valid_indices = np.all(z_scores < 3, axis=1)                                                            
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
    # Handle missing data with mean imputation                                                              
    imputer = SimpleImputer(strategy='mean')                                                                
    data_points = imputer.fit_transform(data_points)                                                        
                                                                                                            
    # Perform feature scaling/normalization                                                                 
    scaler = StandardScaler()                                                                               
    data_points = scaler.fit_transform(data_points)                                                         
                                                                                                            
    # Dimensionality reduction with PCA                                                                     
    pca = PCA(n_components)  # Reduce to 10 principal components                                            
    data_points = pca.fit_transform(data_points)                                                            
                                                                                                            
    # Step 4: Split the dataset into training and validation subsets                                        
    X_train, X_val, y_train, y_val = train_test_split(data_points, labels, test_size=0.2, random_state=42)  #just keep this value here because it complains if its not initialised 
                                                                                                            
    return X_train, X_val, y_train, y_val                                                                   
                                                                                                            
# Example usage                                                                                             
n_components = 6                                                                                            
random_state = 10                                                                                           
data_file = 'traindata.txt'                                                                                 
labels_file = 'trainlabels.txt'                                                                             
X_train, X_val, y_train, y_val = preprocess_data(data_file, labels_file, n_components, random_state)        
#print(X_val)                                                                                               
                                                                                                            
