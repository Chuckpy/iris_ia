import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils


from sklearn.preprocessing import LabelEncoder

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values


labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)


classificador = Sequential()
    #primeira camada oculta
classificador.add(Dense(units = 4, activation= "tanh",
                        kernel_initializer = "random_uniform", input_dim = 4 ))    
 #Dropout para zerar aleatoriamente uma porcentagem de entrada 
classificador.add(Dropout(0.2))

    #segunda camada oculta
classificador.add(Dense(units = 4, activation= "tanh",
                        kernel_initializer = "random_uniform" ))
    #segundo Dropout feito em camada oculta
classificador.add(Dropout(0.2))

    #output camada de saida
classificador.add(Dense(units= 3, activation ="softmax"))

classificador.compile(optimizer= "adam", loss = "sparse_categorical_crossentropy",
                    metrics = ["accuracy"])

classificador.fit(previsores,classe, batch_size=(10), epochs= 500)

#salvando os parametros da ia 
classificador_json = classificador.to_json()
with open("iris.json", "w") as json_file :
    json_file.write(classificador_json)

#salvando os pesos da ia    
classificador.save_weights ("iris.h5")    