import numpy as np
from math import sqrt
from queue import PriorityQueue
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split #separando em dados teste e treinamento 
from collections import namedtuple
from operator import itemgetter
from pprint import pformat
from math import floor
import pandas as pd
import random

class KDNode(object):
    def __init__(self,value,label,left,right,depth):
        self.value = value
        self.label =label
        self.left = left
        self.right = right
        self.depth = depth

class arvoreKD(object):
    def __init__(self):
          self.root = KDNode(None,None,None,None,None)
          self.root = None

    def build_KDTree(self,points,depth):
      if len(points)==0:
        return None
      if len(points)==1:
        node=KDNode(points,0,None,None,depth)
        return node
      else:

        if depth % 2 ==0:
          k= len(points)
          axis = depth % k
          points.sort(key=itemgetter(0))
          median_idx = len(points)// 2
          p=points[median_idx][0]
        else:
          k= len(points)
          axis = depth % k
          points.sort(key=itemgetter(1))
          median_idx = len(points)// 2
          p=points[median_idx][1]
      
        vleft = self.build_KDTree(points[:median_idx],depth+1)  
        vright = self.build_KDTree(points[median_idx:],depth+1)
  
        node=KDNode(0,p,vleft,vright,depth)
        
        return node

class x_NN(object):
   
    def dist(self,point1,point2):
       
        p1x=point1[0][0]
        p2x=point2[0]
        p1y=point1[0][1]
        p2y=point2[1]
        res=np.sqrt(((p2x-p1x)**2)+((p2y-p1y)**2))
      
        return np.sqrt(((p2x-p1x)**2)+((p2y-p1y)**2))

    def find_nearest_neighbor(self, query, node, best_node, best_distance,axis,best_distances,best_neighbors):

        good_side =None
        bad_side =None

        if node==None:
            return best_neighbors
    
        if node.label==0: #quer dizer que achei o ponto na folha da arvore ccom label 0
            d=self.dist(node.value,query)      # calculo a distancia entre os pontos
            best_neighbors.append((node.value, d))
        
            if  d < best_distance:
                best_node = node
                best_distance = d

        else:
        #decidindo os lados da pesquina na arvore
            if node.depth %2 == 0: #verifico em qual eixo comparar com o label no no da arvore
                axis=0
                if query[0]<node.label:#node.label > query[0]:
                    good_side =node.left
                    bad_side =node.right
                else:
                    good_side = node.right
                    bad_side = node.left
      
            else:
                axis=1
                if query[1]<node.label: #node.label > query[1]:
                    good_side =node.left
                    bad_side =node.right
                else:
                    good_side = node.right
                    bad_side = node.left
        best_neighbors = self.find_nearest_neighbor(query, good_side, best_node, best_distance,axis,best_distances,best_neighbors)
    
        #verifica se vale a pena explorar o outro lado do corte
        if abs(node.label-query[axis]) < best_distance:
            best_neighbors = self.find_nearest_neighbor(query, bad_side, best_node, best_distance,axis,best_distances,best_neighbors)
    
        return best_neighbors
  
    
    def get_metrics(self, testSet, predictions):
        metrics = [0.0, 0.0, 0.0, 0.0]
        metrica=[]
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        last_column = len(testSet[0]) - 1
        for i in range(len(testSet)):
            if testSet[i][last_column] == predictions[i][last_column] == 1:
                tp += 1
            elif testSet[i][last_column] == 0 and predictions[i][last_column] == 1:
                fp += 1
            elif testSet[i][last_column] == 1 and predictions[i][last_column] == 0:
                fn += 1
            elif testSet[i][last_column] == 0 and predictions[i][last_column] == 0:
                tn += 1

        metrics[0] = (tp + tn) / (tp + tn + fp + fn)  # accuracy
      
        if tp+fp != 0:
            metrics[1] = tp / (tp + fp)  # precision
        if tp+fn != 0:
         metrics[2] = tp / (tp + fn)  # recall
  
        return metrics
    
    def get_Neighbors(self,trainingSet, testSet, x): #Como não foi expecificado no trabalho o quantidade de vizinhos mais prox , vamos considerar x=3
        best_neighbors=[]
        neighbors=[]
        best_distances=[]
        best_distance1=[]
        predictions=[]
        lista_aux=[]
        metricas=[]
        tree=arvoreKD()
        node=tree.build_KDTree(trainingSet,0)
        for i in range(len(testSet)):
            best_neighbors = self.find_nearest_neighbor(testSet[i],node,None,np.infty,None,best_distances,best_distance1)
            best_neighbors.sort(key=itemgetter(1))
            for j in range(x): #seleciona os x vizinhos mais prox
                neighbors.append(best_neighbors[j][0])
        
            print("Ponto A")
            print(testSet[i])
            print("Os x vizinhos mais proximos")
            print(neighbors)
            print("Metricas da classe do ponto A")
            #metricas=self.get_metrics()
            #print('Accuracy:', metricas[0])
            #print('Precision:', metricas[1])
            #print('Recall:', metricas[2])
            neighbors.clear()

def get_data2():
  trainSet = []
  testSet = []
  for i in range(0,1000):
    p1 = random.randint(1,100)
    p2 = random.randint(1,100)
    trainSet.append((p1,p2))
  for i in range(0,1000):
    p1 = random.randint(1,100)
    p2 = random.randint(1,100)
    testSet.append((p1,p2))
  return trainSet, testSet
 


def main():
   
    #Aqui fazemos os testes .Fazemos 10 testes com bases aleatorias divindo em conjunto de treino e teste
    for i in range(10):
      x1, y1=get_data2()
      x_treino, x_teste, y_treino,y_teste  = train_test_split(x1, y1, test_size = 0.25,random_state = 1)
      fit=x_NN()
      fit.get_Neighbors(x_treino,y_teste,3)
      
    
  
  
if __name__ == "__main__":
    main()
  

