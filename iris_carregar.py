import numpy as np
import pandas as pd
from keras.models import model_from_json

#abrindo o arquivo de classificação com a ia e os melhores parametros
arquivo = open("iris.json", "r")

estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
#carregando os pesos 
classificador.load_weights("iris.h5")

#novo dado usando a ia ja treinada
novo = np.array([[5.4,3.4,1.7,0.2]])

previsao = classificador.predict(novo)
previsao = (previsao > 0.5)

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

classificador.compile(loss = "categorical_crossentropy", 
                      optimizer = "adam", metrics = ["accuracy"])

#resultado da previsão com base nos dados originais

resultado = classificador.evaluate(previsores,classe)