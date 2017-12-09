package projetosweka;

import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

// Comparar a precisão/acurácia/taxa de erro/revocação do algoritmo
// LeaveOneOut -> Deixa um exemplo de fora e usa o resto para treinar o classificador

public class LeaveOneOutComparator 
{    
    public static void main(String[] args) 
    {
        try
        {
            String file = "/home/marcelo/NetBeansProjects/ProjetosWeka/data/iris.arff";
            
            DataSource ds = new DataSource(file);
            
            Instances ins = ds.getDataSet(); // Método que armazena o dataset em uma variável
            
            int iNumAttributes = ins.numAttributes(); // Número de atributos
            int iNumInstances = ins.numInstances(); // Número de instâncias
            int iClass = iNumAttributes - 1; // Atributo classe
            
            ins.setClassIndex(iClass);
            
            System.out.println("Real;VMP;KNN(3);KNN(7)");
            
            IBk vmp = new IBk();
            IBk knn3 = new IBk(3);
            IBk knn7 = new IBk(7);
            
            Instances baseTreino;
            Instances baseTeste;
            Instance teste;
            
            for(int i = 0; i < iNumInstances; i++) // Leave One Out, 0 a 149
            {
                baseTreino = ins.trainCV(iNumInstances, i);
                baseTeste = ins.testCV(iNumInstances, i);
                
                vmp.buildClassifier(baseTreino);
                knn3.buildClassifier(baseTreino);
                knn7.buildClassifier(baseTreino);
                
                teste = baseTeste.instance(0);
                
                System.out.print(teste.stringValue(iClass) + ";");
                
                teste.setClassMissing();
                
                teste.setClassValue(vmp.classifyInstance(teste));
                
                System.out.print(teste.stringValue(iClass) + ";");
                
                teste.setClassMissing();
                
                teste.setClassValue(knn3.classifyInstance(teste));
                
                System.out.print(teste.stringValue(iClass) + ";");
                
                teste.setClassMissing();
                
                teste.setClassValue(knn7.classifyInstance(teste));
                
                System.out.println(teste.stringValue(iClass));
            }
        }
        catch(Exception e) { System.out.println(e.getMessage()); }
    }
}
