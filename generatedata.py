import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import boto3

class GenerateData:
    data=[]
    scaler = StandardScaler()
    X_train=[]
    X_test=[]
    y_train=[]
    y_test=[]
    def __init__(self, datafile_path):
        print(datafile_path)
        GenerateData.data = pd.read_csv(datafile_path)  
        GenerateData.data.fillna(0, inplace = True)
        #print(GenerateData.data.head())
        ## One-hot encode 'cbwd'
        temp = pd.get_dummies(GenerateData.data['cbwd'], prefix='cbwd')
        GenerateData.data = pd.concat([GenerateData.data, temp], axis = 1)
        del GenerateData.data['cbwd'], temp

        #standardize 
    
        GenerateData.data[['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']] = GenerateData.scaler.fit_transform(GenerateData.data[['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']])

        #print(GenerateData.data.head())
    
        ## Split into train and test - I used the last 1 month data as test, but it's up to you to decide the ratio
        df_train = GenerateData.data.iloc[:(-31*24), :].copy()
        df_test = GenerateData.data.iloc[-31*24:, :].copy()

        ## take out the useful columns for modeling - you may also keep 'hour', 'day' or 'month' and to see if that will improve your accuracy
        GenerateData.X_train = df_train.loc[:, ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir', 'cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv']]#.values.copy()
        GenerateData.X_test = df_test.loc[:, ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir', 'cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv']]#.values.copy()
        GenerateData.y_train = df_train['pm2.5'].values.copy().reshape(-1, 1)
        GenerateData.y_test = df_test['pm2.5'].values.copy().reshape(-1, 1)

    def getTrainingSample(self,batch_size, input_seq_length, output_seq_length):
        x_data = GenerateData.X_train
        y_data = GenerateData.y_train
        input_batches = []
        output_batches = []
        n_starting_indexes = len(x_data) / (input_seq_length + output_seq_length)

        for i in range(batch_size):
            starting_index = np.random.randint(0,n_starting_indexes)
            starting_index_offset = starting_index * (input_seq_length + output_seq_length)

            an_input_batch_y = x_data[starting_index_offset: starting_index_offset + input_seq_length]
            # print('an input batch: ' , an_input_batch_y)
            input_data = an_input_batch_y[['pm2.5','DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir', 'cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv']]
            # print(input_data)
            input_batches.append(np.array(input_data))

            an_output_batch_y = y_data[starting_index_offset + input_seq_length:starting_index_offset + input_seq_length + output_seq_length]
            # print('an output batch',an_output_batch_y)
            #output_data = an_output_batch_y[['pm2.5']]
            output_batches.append(np.array(an_output_batch_y))
            #input_batches = np.array(input_batches).reshape(batch_size, input_seq_length, 3)
            #print(input_batches)
            #print(np.array(input_batches)).reshape(batch_size, input_seq_length, 3)
        return input_batches, output_batches

    def getTestSample(self, input_seq_length, output_seq_length,offset):
        input_batches = []
        #output_batches =[]
        the_input_batch = GenerateData.X_test[offset:offset + input_seq_length]
        the_batch_data = the_input_batch[['pm2.5','DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir', 'cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv']]
        input_batches.append(np.array(the_batch_data))

        the_output_batch = np.array(GenerateData.y_test[offset+input_seq_length: offset+ input_seq_length +  output_seq_length])

        #print(the_output_batch)
        #outputs= np.array([GenerateData.y_train[offset + input_seq_length: offset + input_seq_length + output_seq_length]])

        return input_batches, the_output_batch


    def reshape(self, input_array, sequence_length, input_dimension):
        reshaped = [None]* sequence_length 
        for t in range(sequence_length):
            x = input_array[:,t].reshape(-1,input_dimension)
            reshaped[t]=x
        return(np.array(reshaped))
    
    def plot(self):
        cols_to_plot = ["pm2.5", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
        i = 1
        # plot each column
        plt.figure(figsize = (10,12))
        for col in cols_to_plot:
            plt.subplot(len(cols_to_plot), 1, i)
            plt.plot(GenerateData.X_train[col])
            plt.title(col, y=0.5, loc='left')
            i += 1
        plt.show()



#test


#bucket='culturehub'
#data_key = 'seqtoseq/PRSA_data_2010.1.1-2014.12.31.csv'
#data_location = 's3://{}/{}'.format(bucket, data_key)
#gd = GenerateData(data_location)
#input_batches, output_batches = gd.getTrainingSample(1,30,5)
#print(input_batches)
#print(output_batches)
#gd.plot()